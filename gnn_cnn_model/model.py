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
    def __init__(self, n_nodes, n_paths, walk_length, n_label, dim, feature,
                 d_model, d_inner, d_k, d_v, n_head, n_layers, pretrain=False, p_drop=0.1, n_position=5000):
        super(Transformer, self).__init__()
        self.pretrain = pretrain
        self.n_words = feature.shape[1]
        if pretrain:
            # self.node_emb = nn.Embedding(n_nodes, dim)
            # self.node_emb.weight.data = feature
            self.dim = d_model
            # self.linear = nn.Linear(dim, d_model)
            self.feature = feature
            self.W = nn.Parameter(torch.empty(size=(self.n_words, d_model)), requires_grad=True)
            nn.init.xavier_uniform_(self.W.data, gain=1.414)
        else:
            self.node_emb = nn.Embedding(n_nodes, d_model)
            nn.init.xavier_normal_(self.node_emb.weight.data)
            self.dim = d_model
        self.conv = convnet(n_paths, d_model).to(feature.device)

        # self.mlp_layers = nn.Sequential(
        #     nn.Dropout(p=p_drop),
        #     nn.Linear(d_model*4, d_model*2),
        #     nn.ReLU(),
        #     nn.Dropout(p=p_drop),
        #     nn.Linear(d_model*2, d_model)
        # )

        self.encoder = Encoder(self.dim, d_model, d_inner, d_k, d_v, n_head, n_layers, p_drop, n_position)
        self.attn = nn.Linear(d_model * 2, 2)
        self.predict_layer = nn.Linear(d_model, n_label)

        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.tau = 1

    def forward(self, targets, path_tensor_1, path_tensor_2):
        # tensor emb
        if self.pretrain:
            path_tensor_1 = F.embedding(path_tensor_1, self.feature)
            path_tensor_emb_1 = torch.matmul(path_tensor_1, self.W)
            path_tensor_2 = F.embedding(path_tensor_2, self.feature)
            path_tensor_emb_2 = torch.matmul(path_tensor_2, self.W)

            target_tensor = F.embedding(targets, self.feature)
            target_emb = torch.matmul(target_tensor, self.W)
        else:
            path_tensor_emb_1 = self.node_emb(path_tensor_1) # nodes x path x length
            path_tensor_emb_2 = self.node_emb(path_tensor_2)

            target_emb = self.node_emb(targets)
        # target_emb = self.mlp_layers(target_emb)
        conv_emb_1 = self.conv(path_tensor_emb_1).squeeze(1)
        conv_emb_2 = self.conv(path_tensor_emb_2).squeeze(1)

        attended_1 = self.trans_emb(target_emb, conv_emb_1)
        attended_2 = self.trans_emb(target_emb, conv_emb_2)

        same = torch.exp(self.cos_sim(attended_1, attended_2)/self.tau)

        sim = sim_matrix(attended_1, attended_1, tau=1).detach()
        # sim_2 = self.sim_matrix(conv_emb_2, conv_emb_2)
        diff = torch.sum(sim, dim=1)
        tmp = same/diff
        ssl = -torch.log(same/diff)
        score = self.predict_layer(attended_1)
        return score, ssl

    def trans_emb(self, target_emb, conv_emb):
        enc_emb, *_ = self.encoder(conv_emb, target_emb)
        enc_emb = enc_emb.squeeze()

        combined = torch.cat((target_emb, enc_emb), dim=1)
        combined = torch.tanh(combined)
        score = F.softmax(self.attn(combined), dim=1)
        score_t = score[:, 0].unsqueeze(1)
        score_i = score[:, 1].unsqueeze(1)
        attended = score_t * target_emb + score_i * enc_emb
        return attended


def sim_matrix(a, b, tau=1, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    sim_mt = sim_mt/tau
    sim_mt = torch.exp(sim_mt)
    sim_mt = sim_mt.fill_diagonal_(0)
    return sim_mt


class convnet(nn.Module):
    def __init__(self, in_channel, dim):
        super(convnet, self).__init__()
        # self.conv1a = nn.Conv2d(in_channel, 16, kernel_size=(3, 3), padding=1)
        # self.conv1b = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1)
        # self.pool1 = nn.MaxPool2d(2, 2)
        # self.conv2a = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        # self.conv2b = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1)
        # self.pool2 = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # x = self.conv1a(x)
        # x = torch.tanh(x)
        # x = self.conv1b(x)
        # x = torch.tanh(x)
        # x = self.pool1(x)
        #
        # x = self.conv2a(x)
        # x = torch.tanh(x)
        # x = self.conv2b(x)
        # x = torch.tanh(x)
        # x = self.pool2(x)

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
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
        x = x + self.pe[:, 1:x.size(1)+1].clone().detach()
        return self.dropout(x)




