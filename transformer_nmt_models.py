
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint



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


class TransformerSeq2Seq(nn.Module):

    def __init__(self, ntoken, ninp, nhid, nhead, nlayers,
        tie_encoder_decoder=True,
        dropout=0.1,
        bptt=256,
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

        self.pos_encoder = PositionalEncoding(ninp, dropout, max_len=self.bptt)

        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)

        self.transformer = Transformer(ninp, nhead, \
            custom_encoder=self.transformer_encoder, \
            custom_decoder=self.transformer_decoder
        )

        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.nopeek_mask = None

        self.init_weights()

    def _create_mask(self, input_lengths):
        return (torch.arange(self.bptt).unsqueeze(0).to(input_lengths.device) >= input_lengths.unsqueeze(1))

    def _create_nopeek_mask(self, length, device):
        mask = (torch.triu(torch.ones(length, length)) == 1).t()
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_len, trg, trg_mask=None, trg_key_padding_mask=None, **kwargs):
        src_key_mask = self._create_mask(src_len)
        trg_key_mask = trg_key_padding_mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)

        trg = self.encoder(trg) * math.sqrt(self.ninp)
        trg = self.pos_encoder(trg)

        output = self.transformer(src, trg, \
            tgt_mask=trg_mask, \
            src_key_padding_mask=src_key_mask, \
            tgt_key_padding_mask=trg_key_mask, \
            memory_key_padding_mask=src_key_mask.clone()
        )

        return self.decoder(output)