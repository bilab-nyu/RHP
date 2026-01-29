import torch
import torch.nn as nn
import scipy
import math
import os
import numpy as np
import torch.nn.functional as F
from common import AddAndNorm, GLU


    
class FoePredict_risky_time(nn.Module):
    def __init__(self, args):
        super(FoePredict_risky_time, self).__init__()
        self.lstm_encoder_size = args['lstm_encoder_size']
        self.in_length = args['in_length']
        self.endpoint_time = args['endpoint_time']

        self.goal_pred = nn.Linear(self.lstm_encoder_size* self.in_length, self.endpoint_time*4)


    def forward(self, grid_feature, adj, values):
        batch_size, N, timesteps, channels = grid_feature.shape
        values = values.view(batch_size, N, -1).contiguous()
        foe_goal = self.goal_pred(values)
        foe_goal = foe_goal.view(batch_size, N, self.endpoint_time, 4).contiguous()
        ref_pos = grid_feature[:, :, -1:, 9:11].repeat(1, 1, self.endpoint_time, 1)
        # scaling = torch.ones_like(foe_goal[:, :, :, 0])
        # scaling[:] = torch.arange(1, 6).view(1, 1, 5)
        foe_goal[:, :, :, 0] = foe_goal[:, :, :, 0]*20 + ref_pos[:, :, :, 1] * 0.3048
        foe_goal[:, :, :, 1] = foe_goal[:, :, :, 1]*5 + ref_pos[:, :, :, 0] * 0.3048
        foe_goal = foe_goal * adj.unsqueeze(2).unsqueeze(3).repeat(1, 1, self.endpoint_time, 4)
        foe_goal[:, :, :, 2] = foe_goal[:, :, :, 2]*20
        foe_goal[:, :, :, 3] = foe_goal[:, :, :, 3]*5

        return foe_goal
    
class FoePredict_highd_time(nn.Module):
    def __init__(self, args):
        super(FoePredict_highd_time, self).__init__()
        self.lstm_encoder_size = args['lstm_encoder_size']
        self.in_length = args['in_length']
        self.endpoint_time = args['endpoint_time']

        self.goal_pred = nn.Linear(self.lstm_encoder_size* self.in_length, self.endpoint_time*4)


    def forward(self, grid_feature, adj, values):
        batch_size, N, timesteps, channels = grid_feature.shape
        values = values.view(batch_size, N, -1).contiguous()
        foe_goal = self.goal_pred(values)
        foe_goal = foe_goal.view(batch_size, N, self.endpoint_time, 4).contiguous()
        ref_pos = grid_feature[:, :, -1:, 9:11].repeat(1, 1, self.endpoint_time, 1)
        foe_goal[:, :, :, 0] = foe_goal[:, :, :, 0]*10 + ref_pos[:, :, :, 1] * 0.3048
        foe_goal[:, :, :, 1] = foe_goal[:, :, :, 1] + ref_pos[:, :, :, 0] * 0.3048
        foe_goal = foe_goal * adj.unsqueeze(2).unsqueeze(3).repeat(1, 1, self.endpoint_time, 4)
        foe_goal[:, :, :, 2] = foe_goal[:, :, :, 2]*10

        return foe_goal
    
