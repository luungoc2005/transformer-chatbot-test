import torch
import torch.nn.functional as F

from os import path
from utils import get_default_tokenizer

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


class BeamHypotheses(object):
    def __init__(self, n_hyp, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty


def launch(model_params, checkpoint_path, device='cuda'):
    print('model_params:\t', model_params)
    
    max_length = model_params['bptt']

    tokenizer = get_default_tokenizer()

    pad_token = tokenizer.token_to_id('[PAD]')
    eos_token = tokenizer.token_to_id('[SEP]')
    eod_token = tokenizer.token_to_id('[DOC_SEP]')
    p0_token = tokenizer.token_to_id('[P0]')
    vocab_size = tokenizer._tokenizer.get_vocab_size()

    assert eos_token is not None, 'Invalid tokenizer files - EOS token cannot be null'

    # Model
    
    # from models import TransformerModel, LSTMModel
    from lstm_nmt_models import LSTMSeq2Seq
    from transformer_nmt_models import TransformerSeq2Seq
    
    model_type = model_params.get('model_type', 'transformer')
    assert model_type in ['transformer', 'lstm']

    if model_type == 'transformer':
        model = TransformerSeq2Seq(ntoken=vocab_size, src_pad_idx=pad_token, **model_params)
    else:
        model = LSTMSeq2Seq(ntoken=vocab_size, src_pad_idx=pad_token, **model_params)
    
    if checkpoint_path and path.exists(checkpoint_path):
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint_state = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint_state)

    model = model.to(device)

    @torch.no_grad()
    def _generate(
        input_ids=None,
        max_length=max_length,
        do_sample=True,
        num_beams=5,
        temperature=1.3,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.2,
        eos_token_ids=[eos_token, eod_token, pad_token],
        length_penalty=1.0,
        num_return_sequences=1,
        vocab_size=vocab_size
    ):
        model.eval()

        batch_size = 1
        cur_len = 1

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=False) for _ in range(batch_size)
        ]

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # cache compute states
        past = None

        # done sentences
        done = [False for _ in range(batch_size)]

        trg = torch.zeros((max_length, batch_size)).long().fill_(pad_token)
        trg[0,:] = p0_token

        src = input_ids.t()
        src_length = torch.LongTensor([min(max_length, src.size(0))] * batch_size).to(device)

        model_outputs = model(
            src, 
            src_length,
            trg,
            teacher_forcing_ratio=0
        )
        model_outputs = torch.softmax(model_outputs, dim=-1)
        decoded = torch.max(model_outputs, dim=-1)[1].t()

        return decoded


    model_input = ''

    while True:
        user_prompt = input(' >>> ')

        if user_prompt == 'exit':
            exit()

        else:
            model_input += ' [P0] ' + user_prompt + ' [SEP] [P1] '

            encoded = tokenizer.encode(model_input)
            input_ids = encoded.ids
            input_ids = torch.LongTensor(input_ids).unsqueeze(0)
            input_ids = input_ids.to(device)

            output = _generate(input_ids=input_ids, max_length=min(max_length, input_ids.size(1) + 40))

            # if num_return_sequences != 1:
            #     output = output.view(batch_size, num_return_sequences, -1)

            output = output[0].cpu().tolist()
            response = tokenizer.decode(output, skip_special_tokens=False)

            eod_token = '[DOC_SEP]'

            if eod_token in response:
                response = response[response.index(eod_token):]

            start_token = '[P1]'
            sep_token = '[SEP]'

            if start_token in response:
                start_idx = response.index(start_token) + len(start_token) + 1
                response = response[start_idx:]
            
            if sep_token in response:
                sep_idx = response.index(sep_token)
                response = response[:sep_idx]
            
            model_input += response + f' {sep_token} '

            print('Bot: ' + response)

if __name__ == '__main__':
    params_path = './models/model_params.json'
    checkpoint_path = './models/model_state.pt'

    import json
    with open(params_path, 'r') as params_fp:
        model_params = json.load(params_fp)
    
    launch(model_params, checkpoint_path, device='cpu')
