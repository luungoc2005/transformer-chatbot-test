
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint


# https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html
class TiedTransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers
    "Tied": ALBERT-like sharing of parameters across layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TiedTransformerEncoder, self).__init__()
        self.layer = encoder_layer
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
            r"""Pass the input through the encoder layers in turn.

            Args:
                src: the sequnce to the encoder (required).
                mask: the mask for the src sequence (optional).
                src_key_padding_mask: the mask for the src keys per batch (optional).

            Shape:
                see the docs in Transformer class.
            """
            output = src

            for i in range(self.num_layers):
                output = self.layer(output, src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask)

            if self.norm:
                output = self.norm(output)

            return output

class TiedTransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TiedTransformerDecoder, self).__init__()
        self.layer = decoder_layer
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for i in range(self.num_layers):
            output = self.layer(output, memory, tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output

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
        tie_layers=True,
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
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers) \
            if not tie_layers \
            else TiedTransformerEncoder(encoder_layers, nlayers)

        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers) \
            if not tie_layers \
            else TiedTransformerDecoder(decoder_layers, nlayers)

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