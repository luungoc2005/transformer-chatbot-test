
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from custom_layers import \
    TiedTransformerEncoder, \
    TiedTransformerDecoder, \
    RZTXTransformerEncoderLayer, \
    RZTXTransformerDecoderLayer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEmbeddings(nn.Module):

    def __init__(self, ntoken, emb_size, ninp, max_len=5000):
        super(TransformerEmbeddings, self).__init__()

        self.ninp = ninp
        
        # self.pos_encoder = PositionalEncoding(emb_size, dropout, max_len=max_len)
        self.pos_emb = nn.Embedding(max_len, emb_size)
        self.encoder = nn.Embedding(ntoken, emb_size)

        if emb_size != ninp:
            self.emb_norm = nn.LayerNorm(emb_size)
            self.emb_linear = nn.Linear(emb_size, ninp)
        else:
            self.emb_norm = None
            self.emb_linear = None

    def forward(self, x):
        # x = self.encoder(x) * math.sqrt(self.ninp)
        # x = self.pos_encoder(x)
        seq_length = x.size(0)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(1).expand(x.size())

        x = self.encoder(x) + self.pos_emb(position_ids)

        if self.emb_linear is not None:
            x = self.emb_norm(x)
            x = self.emb_linear(x)

        return x

class TransformerSeq2Seq(nn.Module):

    def __init__(self, ntoken, emb_size, ninp, nhid, nhead, nlayers,
        tie_encoder_decoder=True,
        tie_layers=True,
        dropout=0.1,
        bptt=256,
        initializer_range=.02,
        **kwargs
    ):
        super(TransformerSeq2Seq, self).__init__()
        from torch.nn import \
            Transformer, \
            TransformerEncoder, \
            TransformerEncoderLayer, \
            TransformerDecoder, \
            TransformerDecoderLayer
        self.model_type = 'Transformer'
        self.bptt = bptt
        self.initializer_range = initializer_range

        self.embedding = TransformerEmbeddings(ntoken, emb_size, ninp, max_len=self.bptt)

        encoder_layers = RZTXTransformerEncoderLayer(ninp, nhead, nhid, dropout, activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers) \
            if not tie_layers \
            else TiedTransformerEncoder(encoder_layers, nlayers)

        decoder_layers = RZTXTransformerDecoderLayer(ninp, nhead, nhid, dropout, activation='gelu')
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers) \
            if not tie_layers \
            else TiedTransformerDecoder(decoder_layers, nlayers)

        self.transformer = Transformer(ninp, nhead, \
            custom_encoder=self.transformer_encoder, \
            custom_decoder=self.transformer_decoder, \
            dropout=dropout,
            activation='gelu'
        )

        self.ninp = ninp

        if ninp != emb_size:
            self.dec_linear = nn.Linear(ninp, emb_size)
            self.dec_norm = nn.LayerNorm(emb_size)
        else:
            self.dec_linear = None
            self.dec_norm = None

        self.decoder = nn.Linear(emb_size, ntoken, bias=False)

        self.nopeek_mask = None
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _create_mask(self, input_lengths):
        return (torch.arange(self.bptt).unsqueeze(0).to(input_lengths.device) >= input_lengths.unsqueeze(1))

    def _create_nopeek_mask(self, length, device):
        mask = (torch.triu(torch.ones(length, length)) == 1).t()
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)
        return mask

    def forward(self, src, src_len, trg, trg_mask=None, trg_key_padding_mask=None, **kwargs):
        src_key_mask = self._create_mask(src_len)
        trg_key_mask = trg_key_padding_mask

        src = self.embedding(src)
        trg = self.embedding(trg)

        output = self.transformer(src, trg, \
            tgt_mask=trg_mask, \
            src_key_padding_mask=src_key_mask, \
            tgt_key_padding_mask=trg_key_mask, \
            memory_key_padding_mask=src_key_mask.clone()
        )

        if self.dec_linear is not None:
            output = self.dec_linear(output)

        output = F.gelu(output)

        if self.dec_norm is not None:
            output = self.dec_norm(output)

        return self.decoder(output)