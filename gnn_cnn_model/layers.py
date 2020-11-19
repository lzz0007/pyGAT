import torch
import torch.nn as nn
from gnn_cnn_model.sublayers import *


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input, target_emb):
        target_emb = target_emb.unsqueeze(1)
        enc_output, enc_attn = self.slf_attn(target_emb, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_attn