import math

import torch as t
from torch import nn
import torch.nn.functional as F
from common import AddAndNorm, GLU, PositionalEncoding
from einops import rearrange, repeat


class GDEncoder_all(nn.Module):
    def __init__(self, args):
        super(GDEncoder_all, self).__init__()
        self.device = args['device']
        self.lstm_encoder_size = args['lstm_encoder_size']
        self.n_head = args['n_head']
        self.att_out = args['att_out']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.f_length = args['f_length']
        self.relu_param = args['relu']
        self.traj_linear_hidden = args['traj_linear_hidden']
        self.use_maneuvers = args['use_maneuvers']
        self.use_elu = args['use_elu']
        self.use_spatial = args['use_spatial']
        self.dropout = args['dropout']
        # traj encoder
        self.linear1 = nn.Linear(self.f_length, self.traj_linear_hidden)
        self.lstm = nn.LSTM(self.traj_linear_hidden, self.lstm_encoder_size, batch_first=True)
        # self.encoder_layers = 3 # 3
        if args['dataset'] == "highd":
            self.encoder_layers = 1
        elif args['dataset'] == "shrp2":
            self.encoder_layers = 3
        self.encoder_blocks = nn.ModuleList([encoder_block(args) for _ in range(self.encoder_layers)])
        # self.addAndNorm = AddAndNorm(self.lstm_encoder_size)

        if self.use_elu:
            self.activation = nn.ELU()
        else:
            self.activation = nn.LeakyReLU(self.relu_param)


    def forward(self, grid_feature, adj):
        batch_size, N, timesteps, channels = grid_feature.shape
        grid = grid_feature.view(batch_size * N, timesteps, channels).contiguous()
        grid_enc = self.activation(self.linear1(grid))
        grid_hidden_enc, (_, _) = self.lstm(grid_enc)
        grid_hidden_enc = grid_hidden_enc.view(batch_size, N, timesteps, self.lstm_encoder_size).contiguous()
        for i in range(self.encoder_layers):
            grid_hidden_enc = self.encoder_blocks[i](grid_hidden_enc, adj)
            # grid_hidden_enc = self.addAndNorm(grid_hidden_enc1, grid_hidden_enc)


        return grid_hidden_enc


def outputActivation(x):
    muX = x[:, :, 0:1]
    muY = x[:, :, 1:2]
    sigX = x[:, :, 2:3]
    sigY = x[:, :, 3:4]
    rho = x[:, :, 4:5]
    sigX = t.exp(sigX)
    sigY = t.exp(sigY)
    rho = t.tanh(rho)
    out = t.cat([muX, muY, sigX, sigY, rho], dim=2)
    return out


class encoder_block(nn.Module):
    def __init__(self, args):
        super(encoder_block, self).__init__()
        self.lstm_encoder_size = args['lstm_encoder_size']
        self.n_head = args['n_head']
        self.att_out = args['att_out']
        self.in_length = args['in_length']
        self.use_spatial = args['use_spatial']
        self.dropout = args['dropout']

        self.spatial = spatial_transformer(args)
        # self.spatialGAT = spatialGAT_self_all(args)
        self.first_glu = GLU(
            input_size=self.n_head * self.att_out,
            hidden_layer_size=self.lstm_encoder_size,
            dropout_rate=self.dropout)
        self.second_glu = GLU(
            input_size=self.n_head * self.att_out,
            hidden_layer_size=self.lstm_encoder_size,
            dropout_rate=self.dropout)
        #  time attention
        self.d_model = self.n_head * self.att_out
        # self.temporal_embed = nn.Linear(self.lstm_encoder_size, self.d_model)
        self.position_encoding = PositionalEncoding(self.lstm_encoder_size, self.in_length)
        self.qt = nn.Linear(self.lstm_encoder_size, self.d_model)
        self.kt = nn.Linear(self.lstm_encoder_size, self.d_model)
        self.vt = nn.Linear(self.lstm_encoder_size, self.d_model)
        #  addAndNorm
        self.addAndNorm = AddAndNorm(self.lstm_encoder_size)

    def forward(self, grid_hidden_enc, adj):
        # gate
        grid_hidden_enc = grid_hidden_enc.permute(0, 2, 1, 3)
        values = self.spatial(grid_hidden_enc, adj)
        spa_values, _ = self.first_glu(values)
        # values = self.addAndNorm(grid_hidden_enc, spa_values)
        # values = self.spatial(grid_hidden_enc,  adj, risk)
        # spa_values, _ = self.first_glu(values)

        # Residual connection
        values = self.addAndNorm(grid_hidden_enc, spa_values)
        # temporal attention begin---------------------------
        hist_hidden_enc = grid_hidden_enc.permute(0, 2, 1, 3).contiguous()
        values = values.permute(0, 2, 1, 3).contiguous()
        values = self.position_encoding(values)

        qt = t.cat(t.split(self.qt(values), int(self.att_out), -1), 0)
        kt = t.cat(t.split(self.kt(values), int(self.att_out), -1), 0).permute(0, 1, 3, 2)  ####
        vt = t.cat(t.split(self.vt(values), int(self.att_out), -1), 0)
        a = t.matmul(qt, kt)
        a /= math.sqrt(self.lstm_encoder_size)
        a = t.softmax(a, -1)
        values = t.matmul(a, vt)
        values = t.cat(t.split(values, int(hist_hidden_enc.shape[0]), 0), -1)
        # ------------------------
        # gate
        time_values, _ = self.second_glu(values)
        # Residual connection,
        if self.use_spatial:
            spa_values = spa_values.permute(0, 2, 1, 3).contiguous()
            values = self.addAndNorm(hist_hidden_enc, spa_values, time_values)
        else:
            values = self.addAndNorm(hist_hidden_enc, time_values)
        return values



class spatial_transformer(nn.Module):
    def __init__(self, args):
        super(spatial_transformer, self).__init__()
        self.n_head = args['n_head']
        self.att_out = args['att_out']
        self.d_model = self.att_out * self.n_head
        self.lstm_encoder_size = args['lstm_encoder_size']
        self.qff = nn.Linear(self.lstm_encoder_size, self.d_model)
        self.kff = nn.Linear(self.lstm_encoder_size, self.d_model)
        self.vff = nn.Linear(self.lstm_encoder_size, self.d_model)
        self.ffn = nn.Linear(self.d_model, self.lstm_encoder_size)

    def forward(self,  x, adj):
        batch_size, timesteps, N, channels = x.shape

        qt = t.cat(t.split(self.qff(x), self.att_out, -1), 0)
        kt = t.cat(t.split(self.kff(x), self.att_out, -1), 0).permute(0, 1, 3, 2)
        vt = t.cat(t.split(self.vff(x), self.att_out, -1), 0)
        a = t.matmul(qt, kt)
        a /= math.sqrt(self.lstm_encoder_size)
        a = t.softmax(a, -1)
        # adj = adj.unsqueeze(2) * adj.unsqueeze(1)
        # adj = adj.unsqueeze(1).repeat(self.n_head, timesteps, 1, 1)
        #
        # zero_vec = -9e15 * t.ones_like(a)
        # attention = t.where(adj > 0, a, zero_vec)
        # a = t.softmax(attention, -1)

        goal_values = t.matmul(a, vt)
        goal_values = t.cat(t.split(goal_values, batch_size, 0), -1)
        # goal_values = self.ffn(goal_values)
        # goal_values = self.addAndNorm(goal_values, risk_embedding)
        return goal_values

