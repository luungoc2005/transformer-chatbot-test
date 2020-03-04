
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


class TransformerEmbeddings(nn.Module):

    def __init__(self, ntoken, emb_size, ninp, max_len=5000):
        super(TransformerEmbeddings, self).__init__()

        self.ninp = ninp
        
        # self.pos_encoder = PositionalEncoding(emb_size, dropout, max_len=max_len)
        self.pos_emb = nn.Embedding(max_len, emb_size)
        self.encoder = nn.Embedding(ntoken, emb_size)
        self.emb_norm = nn.LayerNorm(emb_size)

        self.emb_linear = nn.Linear(emb_size, ninp)

    def forward(self, x):
        # x = self.encoder(x) * math.sqrt(self.ninp)
        # x = self.pos_encoder(x)
        seq_length = x.size(0)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(1).expand(x.size())

        x = self.encoder(x) + self.pos_emb(position_ids)
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

        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers) \
            if not tie_layers \
            else TiedTransformerEncoder(encoder_layers, nlayers)

        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout, activation='gelu')
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
        self.dec_linear = nn.Linear(ninp, emb_size)
        self.dec_norm = nn.LayerNorm(emb_size)
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

        output = self.dec_linear(output)
        output = F.gelu(output)
        output = self.dec_norm(output)

        return self.decoder(output)