import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from gnn_cnn_model.layers import EncoderLayer


# class recomm(nn.Module):
#     def __init__(self, transformer, num_mlp_layers, input_dim, n_label, dropout=0.1):
#         super(recomm, self).__init__()
#
#         self.transformer = transformer
#
#         self.cosine = nn.CosineSimilarity(dim=1)
#         self.dropout = dropout
#
#         self.input_dim = 800
#         mlp_modules = []
#         for i in range(num_mlp_layers):
#             input_size = int(self.input_dim / (2 ** i))
#             mlp_modules.append(nn.Dropout(p=self.dropout))
#             mlp_modules.append(nn.Linear(input_size, input_size // 2))
#             mlp_modules.append(nn.ReLU())
#         self.mlp_layers = nn.Sequential(*mlp_modules)
#
#         # self.predict_layer = nn.Linear(800, n_label)
#         self.predict_layer = nn.Linear(int(self.input_dim / (2 ** num_mlp_layers)), n_label)
#         self._init_weight_()
#
#     def forward(self, pos, pos_target):
#         pos_emb = self.transformer(pos_target, pos)
#         output_pos_mlp = self.mlp_layers(pos_emb)
#         prediction_pos = self.predict_layer(output_pos_mlp)
#         # prediction_pos = F.softmax(prediction_pos, dim=1)
#         return prediction_pos
#
#     def _init_weight_(self):
#         """ We leave the weights initialization here. """
#         for m in self.mlp_layers:
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#         nn.init.kaiming_uniform_(self.predict_layer.weight,
#                                  a=1, nonlinearity='sigmoid')


class Transformer(nn.Module):
    def __init__(self, n_nodes, n_paths, n_label, dim, feature,
                 d_model, d_inner, d_k, d_v, n_head, n_layers, pretrain=False, p_drop=0.1, n_position=5000):
        super(Transformer, self).__init__()

        if pretrain:
            self.node_emb = nn.Embedding(n_nodes, dim)
            self.node_emb.weight.data = feature
            self.dim = dim
        else:
            self.node_emb = nn.Embedding(n_nodes, d_model)
            nn.init.xavier_normal_(self.node_emb.weight.data)
            self.dim = d_model
        self.conv = convnet(n_paths, n_nodes, self.dim, feature).to(feature.device)
        self.encoder = Encoder(self.dim, d_model, d_inner, d_k, d_v, n_head, n_layers, p_drop, n_position)
        self.attn = nn.Linear(d_model * 2, 2)
        self.predict_layer = nn.Linear(d_model, n_label)

    def forward(self, targets, path_tensor):
        # tensor emb
        path_tensor_emb = self.node_emb(path_tensor)
        conv_emb = self.conv(path_tensor_emb).squeeze()

        # target emb
        target_emb = self.node_emb(targets)
        # target_emb = torch.mm(target_emb, self.T)

        enc_emb, *_ = self.encoder(conv_emb, target_emb)
        enc_emb = enc_emb.squeeze()

        combined = torch.cat((target_emb, enc_emb), dim=1)
        combined = torch.tanh(combined)
        score = F.softmax(self.attn(combined), dim=1)
        score_t = score[:, 0].unsqueeze(1)
        score_i = score[:, 1].unsqueeze(1)
        attended = score_t * target_emb + score_i * enc_emb

        score = self.predict_layer(attended)
        return score


class convnet(nn.Module):
    def __init__(self, in_channel, n_nodes, emb_dim, feature):
        super(convnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 10, kernel_size=1)
        self.conv2 = nn.Conv2d(10, 1, kernel_size=3, padding=1)
        # self.emb = nn.Embedding(n_nodes, emb_dim)
        # nn.init.xavier_normal_(self.emb.weight.data, gain=1.414)
        # self.W = nn.Parameter(torch.empty(size=(emb_dim, 800)).to(feature.device), requires_grad=True)
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # self.emb.weight.data = torch.mm(feature, self.W)

    def forward(self, x):
        # x = self.emb(x)
        # x = torch.mm(x, self.W)
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        return x


class Encoder(nn.Module):
    def __init__(self, dim, d_model, d_inner, d_k, d_v, n_head, n_layers, p_drop=0.1, n_position=5000):
        super(Encoder, self).__init__()

        self.position_enc = PositionalEncoding(dim, p_drop, n_position)
        self.dropout = nn.Dropout(p_drop)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(dim, d_inner, n_head, d_k, d_v, dropout=p_drop) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)

        # self.linear1 = nn.Linear(dim, d_model)
        # self.linear2 = nn.Linear(dim, d_model)

    def forward(self, conv_emb, target_emb, return_attns=False):
        # emb = self.linear(conv_emb)
        enc_output = self.dropout(self.position_enc(conv_emb))
        enc_output = self.layer_norm(enc_output)
        enc_attn_list = []
        for enc_layer in self.layer_stack:
            enc_output, enc_attn = enc_layer(enc_output, target_emb)
            enc_attn_list += [enc_attn] if return_attns else []
        if return_attns:
            return enc_output, enc_attn_list
        return enc_output,


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].clone().detach()
        return self.dropout(x)




