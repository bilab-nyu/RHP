import torch
import torch.nn as nn
import scipy
import math
import os
import numpy as np
import torch.nn.functional as F
from common import *
from torch.distributions import MultivariateNormal, Laplace

def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


    

class transition_encoding(nn.Module):
    def __init__(self, args):
        super(transition_encoding, self).__init__()
        self.n_head = args['n_head']
        self.att_out = args['att_out']
        self.d_model = self.att_out * self.n_head
        self.lstm_encoder_size = args['lstm_encoder_size']
        self.qff = nn.Linear(self.lstm_encoder_size, self.d_model)
        self.kff = nn.Linear(self.lstm_encoder_size, self.d_model)
        self.vff = nn.Linear(self.lstm_encoder_size, self.d_model)
        self.ffn = nn.Linear(self.d_model, self.lstm_encoder_size)

    def forward(self, risk_embedding):
        batch_size, modes, timesteps, channels = risk_embedding.shape
        # risk_embedding = self.lstm(risk_embedding.view(-1, timesteps, channels))[0].view(batch_size, modes, timesteps, -1)
        # risk_embedding = risk_embedding[:,:,1:,:] - risk_embedding[:,:,0:-1,:]
        x0 = risk_embedding[:,:,0:1,:]
        qt = t.cat(t.split(self.qff(x0), self.att_out, -1), 0)
        kt = t.cat(t.split(self.kff(risk_embedding), self.att_out, -1), 0).permute(0, 1, 3, 2)
        vt = t.cat(t.split(self.vff(risk_embedding), self.att_out, -1), 0)
        a = t.matmul(qt, kt)
        a /= math.sqrt(self.lstm_encoder_size)
        a = t.softmax(a, -1)

        goal_values = t.matmul(a, vt).squeeze(2)
        goal_values = t.cat(t.split(goal_values, batch_size, 0), -1)
        goal_values = self.ffn(goal_values)
        return goal_values, t.cat(t.split(a, batch_size, 0), 2)


class ChangePointHead(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, h):
        logits = self.net(h).squeeze(-1)  # (B, T)
        prob = torch.softmax(logits, -1)
        preds = prob.argmax(dim=-1) 
        return prob, preds

class OutputModel(nn.Module):

    def __init__(self, d_k=64):
        super(OutputModel, self).__init__()
        self.d_k = d_k
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.observation_model = nn.Sequential(
            init_(nn.Linear(d_k, d_k)), nn.ReLU(),
            init_(nn.Linear(d_k, d_k)), nn.ReLU(),
            init_(nn.Linear(d_k, 7))
        )
        self.min_stdev = 0.01

    def forward(self, agent_decoder_state, log_std_range=(-1.609, 5.0), rho_limit=0.5):
        T = agent_decoder_state.shape[0]
        BK = agent_decoder_state.shape[1]
        pred_obs = self.observation_model(agent_decoder_state.reshape(-1, self.d_k)).reshape(T, BK, -1)

        x_mean = pred_obs[:, :, 0]
        y_mean = pred_obs[:, :, 1]
        log_std1 = torch.clip(pred_obs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
        log_std2 = torch.clip(pred_obs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
        x_sigma = torch.exp(log_std1)  # (0.2m to 150m)
        y_sigma = torch.exp(log_std2) 
        # x_sigma = F.softplus(pred_obs[:, :, 2]) + self.min_stdev
        # y_sigma = F.softplus(pred_obs[:, :, 3]) + self.min_stdev
        # rho = torch.tanh(pred_obs[:, :, 4]) * 0.9  # for stability
        rho = torch.clip(pred_obs[:, :, 4], min=-rho_limit, max=rho_limit)
        return torch.stack([x_mean, y_mean, x_sigma, y_sigma, rho, pred_obs[:, :, 5], pred_obs[:, :, 6]], dim=2)


class Risk_Decoder_auto_time(nn.Module):
    def __init__(self, args):
        super(Risk_Decoder_auto_time, self).__init__()
        self.ego_goals = []
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.dataset = args['dataset']
        self.endpoint_time = args['endpoint_time']
        self.time_step = args['out_length']//args['endpoint_time']
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        if self.dataset == "highd":
            ego_goals = scipy.io.loadmat(args['ego_goals_path'] + 'ego_goal_down.mat')['C2']
            self.ego_goals.append(ego_goals[:,np.arange(self.time_step-1, self.out_length, self.time_step),:])
            ego_goals = scipy.io.loadmat(args['ego_goals_path'] + 'ego_goal_up.mat')['C2']
            self.ego_goals.append(ego_goals[:,np.arange(self.time_step-1, self.out_length, self.time_step),:])

            self.ego_goals = torch.tensor(np.asarray(self.ego_goals), dtype=torch.float32).to(args['device'])
            self.ego_goals = self.ego_goals[:,:, :,[1, 0, 3, 2]]+1e-4
            ego_goals_r = torch.sqrt(self.ego_goals[:,:,:,0]**2 + self.ego_goals[:,:,:,1]**2)
            ego_goals_theta = torch.atan2(self.ego_goals[:,:,:,1], self.ego_goals[:,:,:,0])
            ego_goals_vr = torch.sqrt(self.ego_goals[:,:,:,2]**2 + self.ego_goals[:,:,:,3]**2)
            ego_goals_vtheta = torch.atan2(self.ego_goals[:,:,:,3], self.ego_goals[:,:,:,2])
            self.ego_goals_rtheta = torch.stack([ego_goals_r, ego_goals_theta, ego_goals_vr, ego_goals_vtheta], dim=-1)
            self.ego_goal_size = self.ego_goals.shape[1]
        elif self.dataset == "shrp2":
            ego_goals = scipy.io.loadmat(args['ego_goals_path']+'/ego_goal_5s.mat')['C2']
            self.ego_goals = torch.tensor(np.asarray(ego_goals[:,np.arange(self.time_step-1, self.out_length, self.time_step),:]), dtype=torch.float32).to(args['device'])
            self.ego_goals = self.ego_goals[:, :,[1, 0, 3, 2,4]]
            self.ego_goal_size = self.ego_goals.shape[0]

        # for filename in os.listdir(args['ego_goals_path']):
        #     if filename.endswith('.mat'):
        #         ego_goals = scipy.io.loadmat(args['ego_goals_path'] + filename)['C']
        #         self.ego_goals.append(ego_goals)    
            
        self.device = args['device']
        self.lstm_encoder_size = args['lstm_encoder_size']
        # self.lstm_encoder_size_decoder = args['lstm_encoder_size_decoder']
        self.n_head = args['n_head']
        self.att_out = args['att_out']
        self.dropout = args['dropout']
        self.c = args['c']
        self.goal_modes = args['goal_modes']
        self.select_endpoint = 1
        self.Q = nn.Parameter(torch.Tensor(self.out_length, 1, self.c, self.lstm_encoder_size), requires_grad=True)
        # self.risk_Q = nn.Parameter(torch.Tensor(1, self.goal_modes, 1, self.lstm_encoder_size), requires_grad=True)
        self.risk_para = nn.Parameter(torch.Tensor(11), requires_grad=True)
        # self.cp_head = ChangePointHead(in_dim=self.lstm_encoder_size, hidden=self.lstm_encoder_size)
        nn.init.xavier_uniform_(self.Q)
        nn.init.zeros_(self.risk_para)
        # nn.init.xavier_uniform_(self.risk_Q)

        self.tx_hidden_size = args['tx_hidden_size']
        if self.dataset == "highd":
            self.num_layers = 1
        elif self.dataset == "shrp2":
            self.num_layers = 2

        self.tx_decoder = []
        self.risk_attn_layers = nn.MultiheadAttention(self.lstm_encoder_size, num_heads=self.n_head, dropout=0.3)
        # self.risk_tran_layers = nn.MultiheadAttention(self.lstm_encoder_size, num_heads=self.n_head, dropout=0.3)
        self.norms = []
        for _ in range(self.num_layers):
            self.tx_decoder.append(nn.TransformerDecoderLayer(d_model=self.lstm_encoder_size, nhead=self.n_head,
                                                              dropout=self.dropout,
                                                              dim_feedforward=self.tx_hidden_size))
        self.tx_decoder = nn.ModuleList(self.tx_decoder)
        self.output_model = OutputModel(d_k=self.lstm_encoder_size)
        # self.transition_model = transition_predict(args)
        self.mlp_prob = nn.Linear(self.out_length, 1)
        self.prob_predictor = nn.Linear(self.lstm_encoder_size, 1)
        if self.dataset == "highd":
            self.riskembed1 = nn.Sequential(init_(nn.Linear(12, self.lstm_encoder_size)))
        elif self.dataset == "shrp2":
            self.riskembed1 = nn.Sequential(init_(nn.Linear(13, self.lstm_encoder_size)))
        self.transition_encode = transition_encoding(args)
        # self.grad_attn = GradientBasedAttention(13, self.lstm_encoder_size,self.lstm_encoder_size)



    def forward(self, values, ego_current, foe_future, grid_feature, goal_stamps=None):
        B = values.shape[0]
        # goal_stamps, probs = self.transition_model(values, foe_future)
        # goal_stamps = torch.ones(B).long().to(self.device)*5
        s_risk_fut, o_risk_fut, ttc_fut, ego_goals = self.get_risk(ego_current, foe_future, grid_feature, goal_stamps)#

        risks =  torch.cat((s_risk_fut, o_risk_fut), dim = -1)
        risks_goals = torch.cat((risks, ego_goals), dim = -1) # [B, modes, time, d_k]
        # risk_transition = self.riskembed1(risks).permute(2, 1, 0, 3).contiguous() # [time, modes, B, d_k]
        # risk_transition = risk_transition.reshape(self.endpoint_time, B * self.goal_modes, -1)
        # risk_tran_seq = self.risk_Q.repeat(1, 1, B, 1).view(1, B * self.goal_modes, -1)
        # ori_risk_embedding = self.risk_tran_layers(query=risk_tran_seq, key=risk_transition, value=risk_transition)[0]
        # ori_risk_embedding = ori_risk_embedding.squeeze(0).reshape(self.goal_modes, B, -1)
        # risk_embedding = ori_risk_embedding.unsqueeze(2).repeat(1, 1, self.c, 1).view(-1, B * self.c, self.lstm_encoder_size)
        

        # risks_goals = risks_goals[:,:,4,:]
        ori_risk_embedding = self.riskembed1(risks_goals)#.reshape(B, self.select_endpoint*self.goal_modes, -1)
        ori_risk_embedding, attn = self.transition_encode(ori_risk_embedding)
        ori_risk_embedding = ori_risk_embedding.permute(1,0,2).contiguous() # [time*modes, B, d_k]
        risk_embedding = ori_risk_embedding.unsqueeze(2).repeat(1, 1, self.c, 1).view(-1, B * self.c, self.lstm_encoder_size)

        ego_soctemp_emb = values.permute(1, 0, 2).contiguous() # [in_length, B, d_k]
        context = ego_soctemp_emb.unsqueeze(2).repeat(1, 1, self.c, 1) 
        context = context.view(-1, B * self.c, self.lstm_encoder_size) # [in_length, B*c, d_k]

        # Decoding
        out_seq = self.Q.repeat(1, B, 1, 1).view(self.out_length, B * self.c, -1)
        time_masks = self.generate_decoder_mask(seq_len=self.out_length, device=self.device)
        for d in range(self.num_layers):
            ego_dec_emb_map = self.risk_attn_layers(query=out_seq, key=risk_embedding, value=risk_embedding)[0]
            out_seq = out_seq + ego_dec_emb_map
            out_seq = self.tx_decoder[d](out_seq, context, tgt_mask=time_masks)
        out_dists = self.output_model(out_seq).reshape(self.out_length, B, self.c, -1).permute(2, 0, 1, 3) # [c, T, B, 5] 

    
        # Mode prediction
        out_seq_prob = out_seq.reshape(self.out_length, B, self.c, -1).permute(2,1,3,0).contiguous()
        mode_params_emb = self.mlp_prob(out_seq_prob).squeeze(-1)
        mode_probs = F.softmax(self.prob_predictor(mode_params_emb).squeeze(-1), dim=0).transpose(0, 1)


        # return  [c, T, B, 5], [B, c]
        output = {}
        output['predicted_probability'] = mode_probs  # #[B, c]
        output['predicted_trajectory'] = out_dists.permute(2, 0, 1, 3)  # [c, T, B, 5] to [B, c, T, 5]
        # output['transition_probs'] = attn
        # output['s_fut_risk'] = s_risk_fut.sum(dim=-1)
        # output['o_fut_risk'] = o_risk_fut.sum(dim=-1)
        # output['ego_current'] = ego_current[:,2]


    
        return output
    
    def generate_decoder_mask(self, seq_len, device):
        ''' For masking out the subsequent info. '''
        subsequent_mask = (torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1)).bool()
        return subsequent_mask
    


    def get_risk(self, ego_current, foe_future, grid_feature, goal_stamp=None):
        batch_size = ego_current.shape[0]
        cur_info = foe_future[:, :, 0:1, :]
        # cur_info = torch.cat([cur_info, torch.zeros(cur_info.shape[0], cur_info.shape[1], cur_info.shape[2], 1).to(self.device)], dim=3)
        foe_future = foe_future[:, 1:, 1:, :]
        if self.dataset == "highd":
            ego_goals1 = self.ego_goals[list(map(int, (ego_current[:, 2]).tolist()))]
            ego_goal_rtheta = self.ego_goals_rtheta[list(map(int, (ego_current[:, 2]).tolist()))]
        elif self.dataset == "shrp2":
            if goal_stamp is not None:
                ego_goals1 = self.ego_goals.permute(1, 0, 2)[goal_stamp - 1]
                ego_goals1 = ego_goals1.unsqueeze(2)
                i_idx = torch.arange(foe_future.shape[0])
                foe_future = foe_future[i_idx, :, goal_stamp - 1, :].unsqueeze(2)
            else:
                ego_goals1 = self.ego_goals.repeat(batch_size, 1, 1, 1)

        # ego_pos_cur = cur_info[:, 0:1, :, 0:2]
        # ego_vel_cur = cur_info[:, 0:1, :, 2:4]
        # foe_pos_cur = cur_info[:, 1:, :, 0:2]
        # foe_vel_cur = cur_info[:, 1:, :, 2:4]

        ego_pos = ego_goals1[:, :, :, 0:2]* 0.3048# + ego_current1
        ego_vel = ego_goals1[:, :, :, 2:4]* 0.3048
        foe_pos = foe_future[:, :, :, 0:2]
        foe_vel = foe_future[:, :, :, 2:4]
        ego_dim = grid_feature[:,0,0,5:7] * 0.3048
        foe_dim = grid_feature[:,1:,0,5:7] * 0.3048

    

        cnt = foe_pos.shape[1]
        # ego_pos_cur = ego_pos_cur.unsqueeze(2).repeat(1,1, cnt,1,1)
        # ego_vel_cur = ego_vel_cur.unsqueeze(2).repeat(1,1, cnt,1,1)
        # foe_pos_cur = foe_pos_cur.unsqueeze(1)
        # foe_vel_cur = foe_vel_cur.unsqueeze(1)
        # ego_dim_cur = ego_dim.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1,1, cnt,1,1)
        # foe_dim_cur = foe_dim.unsqueeze(1).unsqueeze(3)
        # ego_psi_cur = torch.zeros_like(ego_pos_cur[:, :, :, :, 0]) 
        foe_pos_fut = foe_pos.unsqueeze(1).repeat(1,ego_pos.shape[1],1,1,1)
        foe_vel_fut = foe_vel.unsqueeze(1).repeat(1,ego_vel.shape[1],1,1,1)
        ego_pos_fut = ego_pos.unsqueeze(2).repeat(1,1, cnt, 1,1)
        ego_vel_fut = ego_vel.unsqueeze(2).repeat(1,1, cnt, 1,1)
        ego_dim_fut = ego_dim.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1,ego_pos.shape[1], cnt, self.select_endpoint,1)
        foe_dim_fut = foe_dim.unsqueeze(1).unsqueeze(3).repeat(1,ego_pos.shape[1], 1, self.select_endpoint, 1)
        if self.dataset == "risky":
            ego_psi = ego_goals1[:, :, :, 4]
            ego_psi_fut = ego_psi.unsqueeze(2).repeat(1,1, cnt,1)
        else:
            ego_psi_fut = torch.zeros_like(ego_pos_fut[:, :, :, :, 0])

        s_risk_fut, o_risk_fut, ttc_fut = self.cal_risk_xy(ego_pos_fut, ego_vel_fut, ego_dim_fut, foe_pos_fut, foe_vel_fut,
                                               foe_dim_fut, ego_psi_fut)
        # s_risk_cur, o_risk_cur, ttc_cur = self.cal_risk_xy(ego_pos_cur, ego_vel_cur, ego_dim_cur, foe_pos_cur, foe_vel_cur,
        #                                        foe_dim_cur, ego_psi_cur)
        # s_risk_fut = t.concat((s_risk_cur.repeat(1,self.goal_modes,1,1), s_risk_fut), dim=2)
        # o_risk_fut = t.concat((o_risk_cur.repeat(1,self.goal_modes,1,1), o_risk_fut), dim=2)
        # ego_goals1 = t.concat((cur_info[:, 0:1, :, :].repeat(1,self.goal_modes,1,1), ego_goals1), dim=2)

        # s_risk_tran = t.sum(s_risk_fut[:,:,1:,:],dim=3)-t.sum(s_risk_fut[:,:,0:-1,:],dim=3)
        # o_risk_tran = t.sum(o_risk_fut[:,:,1:,:],dim=3)-t.sum(o_risk_fut[:,:,0:-1,:],dim=3)
        # indices = t.arange(s_risk_tran.shape[2], device=s_risk_tran.device).float()
        # # mask_s = (t.abs(s_risk_tran) > 0.1).float()
        # mask_o = (t.abs(o_risk_tran) > 0.1).float()
        # # mean_s = t.abs(s_risk_tran).max(dim=2)[1]+1
        # mean_o = t.abs(o_risk_tran).max(dim=2)[1]+1
        # # mean_s[mask_s.sum(dim=2) == 0] = 5
        # mean_o[mask_o.sum(dim=2) == 0] = 5
        # # goal = t.max(goal_s, goal_o)
        # goal_stamp_o = mean_o.cpu().numpy().astype(int)
        # i_idx = torch.arange(s_risk_fut.shape[0]).unsqueeze(1).expand(-1, s_risk_fut.shape[1])   # (128,100)
        # j_idx = torch.arange(s_risk_fut.shape[1]).unsqueeze(0).expand(s_risk_fut.shape[0], -1)   # (128,100)
        # goal_stamp_o = torch.tensor(goal_stamp_o).to(self.device)
        # s_risk_fut1 = s_risk_fut[i_idx, j_idx, goal_stamp_o, :] 
        # o_risk_fut1 = o_risk_fut[i_idx, j_idx, goal_stamp_o, :] 
        # ego_goals1 = ego_goals1[i_idx, j_idx, goal_stamp_o, :] 
        
        if self.dataset == "shrp2":  
            normalize_goal = torch.tensor([350, 80, 70, 10, 1]).to(self.device)
        elif self.dataset == "highd":
            normalize_goal = torch.tensor([350, 16, 100, 2]).to(self.device)
            # normalize_goal = torch.tensor([350, 1, 70, 1]).to(self.device)

        # ego_goals1 = self.ego_goals[:,4:5,:].unsqueeze(0).repeat(batch_size,1,1,1)
        ego_goals1 = ego_goals1 / normalize_goal
        # one_hot = F.one_hot(goal_stamp.long()-1, num_classes=5).float()
        if goal_stamp is not None:
            ego_goals1 = torch.cat((ego_goals1, goal_stamp.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1,ego_pos.shape[1],1,1)/5), dim=3)
        # ego_goals1 = torch.concat((ego_goals1, goal_stamp_o.unsqueeze(2)), dim=2)

        return s_risk_fut, o_risk_fut,ttc_fut, ego_goals1#.unsqueeze(2).repeat(1,1,cnt,1,1) 



    def cal_risk(self, ego_pos, ego_vel, ego_dim, foe_pos, foe_vel, foe_dim, ego_psi = torch.zeros(1)):
        dx_ori = ego_pos[:, :, :, 0] - foe_pos[:, :, :, 0] #large
        dy_ori = ego_pos[:, :, :, 1] - foe_pos[:, :, :, 1] #small
        if self.dataset == "risky":
            ego_psi = ego_psi.to(ego_pos.device)
            dx = dx_ori*torch.cos(ego_psi)+dy_ori*torch.sin(ego_psi)
            dy = -dx_ori*torch.sin(ego_psi)+dy_ori*torch.cos(ego_psi)
            foe_dim[:, :, :, 0] = 2
        else:
            dx = dx_ori
            dy = dy_ori
        # define an epsilon to prevent 0 distance
        epsilon = 1e-2
        # get dx and dy WITH vehicle shapes
        dx = torch.clamp(torch.abs(dx) - 0.5 * (ego_dim[:, :, :, 0] + foe_dim[:, :, :, 0]), 0) * dx / (abs(dx) + epsilon)
        dy = torch.clamp(torch.abs(dy) - 0.5 * (ego_dim[:, :, :, 1] + foe_dim[:, :, :, 1]), 0) * dy / (abs(dy) + epsilon)

    
        if self.dataset == "risky":
            dVx = -(torch.cos(ego_psi)*(ego_vel[:, :, :, 0] - foe_vel[:, :, :, 0])+
                    torch.sin(ego_psi)*(ego_vel[:, :, :, 1] - foe_vel[:, :, :, 1])) * dx / (abs(dx) + epsilon)
            dVy = -(torch.sin(ego_psi)*foe_vel[:, :, :, 0]-torch.cos(ego_psi)*foe_vel[:, :, :, 1]) * dy / (abs(dy) + epsilon)
        else:
            dVx = -(torch.abs(ego_vel[:, :, :, 0]) - torch.abs(foe_vel[:, :, :, 0])) * dx / (abs(dx) + epsilon)
            dVy = -(ego_vel[:, :, :, 1] - foe_vel[:, :, :, 1]) * dy / (abs(dy) + epsilon)


        vj = torch.abs(ego_vel[:, :, :, 0])
        if self.dataset == "highd":
            gx = vj  #1.5+2
            gy = vj * 0.03 #+1
            bx = vj * 0.04 + 2
            by = 5.5
        elif self.dataset == "risky":
            # gx = torch.abs(ego_vel[:, :, :, :, 0]) *1.5 + 2 #1.5 2
            # gy = torch.abs(ego_vel[:, :, :, :, 1]) *1.5 + 1 #1.5 1
            # bx = torch.abs(ego_vel[:, :, :, :, 0]) * 0.04 + 2#2
            # by = 5.5
            gx = torch.abs(ego_vel[:, :, :, 0]) *(1.5+self.risk_para[0]) + self.risk_para[1]+2  #1.5 2
            gy = torch.abs(ego_vel[:, :, :, 1]) *(1.5+self.risk_para[2])+ self.risk_para[3]+1 #1.5 1
            bx = torch.abs(ego_vel[:, :, :, 0]) *(0.04+self.risk_para[4]) + self.risk_para[5]+2#2
            by = self.risk_para[6] + 5.5
            gx = torch.clamp(gx, min=1)
            gy = torch.clamp(gy, min=1)
            bx = torch.clamp(bx, min=1)
            by = torch.clamp(by, min=1)


        # s_para = self.s_risk_para(ego_vel)
        # s_para = s_para + 1
        # gx = s_para[:, :, :, 0]
        # gy = s_para[:, :, :, 1]

        # get s_risks
        s_risk = torch.exp(-torch.abs(dx / gx) ** bx - torch.abs(dy / gy) ** by) * (foe_pos[:, :, :, 0] != 0).float()
        # s_risk = torch.exp(-2*(torch.abs(dx / gx) ** bx + torch.abs(dy / gy) ** by - 1)) * (foe_pos[:,:,:,:,0] != 0).float()
   

        # _d = 5#3 risky 5  highd 1
        # _t = 7.5#5 risky 7.5 highd 2
        # _b1 = 5#10  5
        # _b2 = 3#2  3
        _d = self.risk_para[7]+5
        _t = self.risk_para[8]+7.5
        _b1 = self.risk_para[9]+5
        _b2 = self.risk_para[10]+3
        _d = torch.clamp(_d, min=1)
        _t = torch.clamp(_t, min=1)
        _b1 = torch.clamp(_b1, min=1)
        _b2 = torch.clamp(_b2, min=1)
        alpha = 1
        beta = 1


        epsilon_v = 1e-4
        # define an epsilon to set negative times to a large number
        epsilon_t = 1e-4
        # get o_risks
        v = torch.sqrt(dVx**2 + dVy**2)
        v_mask = (v != 0) & (dx ** 2 + dy ** 2 > 0)
        # v = torch.clamp(v, epsilon_v)
        dm2 = (dVx * dy - dVy * dx) ** 2 / (dVx**2 + dVy**2 + epsilon_v)
        td = -(dVx * dx + dVy * dy) / (dVx**2 + dVy**2 + epsilon_v)
        td = 1 / torch.clamp(1 / (td+epsilon_t), min=epsilon_t)
        # td = torch.where(td > 1/epsilon_t, 1/epsilon_t, td)
        # td = torch.where(td < 0, 1 / epsilon_t, td)
        o_risk = torch.exp(-alpha * (dm2 / _d) ** _b1 ) * torch.exp(-beta * (td / _t) ** _b2) * v_mask
        # o_risk = torch.exp(-alpha * (dm2 / _d) ** _b1 + 1) * torch.exp(-beta * (td / _t) ** _b2 + 1) * v_mask
        o_risk = o_risk * (foe_pos[:, :, :, 0] != 0).float()


      
        # ttc = self.cal_ttc(ego_pos, ego_vel, foe_pos, foe_vel, ego_dim, foe_dim)

     

        s_risk_sum = torch.sum(s_risk, dim=2)
        o_risk_sum = torch.sum(o_risk, dim=2)
        ttc_sum = []#torch.stack([veh_ttc_x_right_sum, veh_ttc_x_left_sum, veh_ttc_y_up_sum, veh_ttc_y_down_sum], dim=2)

        return s_risk_sum, o_risk_sum, ttc_sum

    def get_final_risk(self, ego_pos,ego_vel, foe_future, grid_feature):
        ego_pos = ego_pos * 0.3048# + ego_current1
        ego_vel = ego_vel *0.3048
        batch_size = ego_pos.shape[0]
        foe_future_n = foe_future
        cnt_foe = foe_future_n.shape[1]
        # cnt_foe = max(len(x) for x in foe_future)
        # padded_lst = [torch.cat([torch.tensor(np.array(x)), torch.zeros(cnt_foe - len(x),self.out_length,4)]) for x in foe_future]
        # foe_future_n = torch.stack(padded_lst).to(self.device)
        cnt_ego = ego_pos.shape[1]
        ego_pos_n = ego_pos.permute(0,2,1,3).reshape(-1,cnt_ego,2)
        ego_vel_n = ego_vel.permute(0,2,1,3).reshape(-1,cnt_ego,2)
        foe_pos = foe_future_n[:, :,:, 0:2].permute(0,2,1,3).reshape(-1,cnt_foe,2)
        foe_vel = foe_future_n[:, :,:, 2:4].permute(0,2,1,3).reshape(-1,cnt_foe,2)
        ego_dim = grid_feature[:,0,0,5:7] * 0.3048
        foe_dim = grid_feature[:,1:,0,5:7] * 0.3048
        # nonzero_rows = (foe_dim != 0).any(dim=2)
        # foe_dim_n = []
        # for i in range(batch_size):
        #     foe_dim_i = foe_dim[i][nonzero_rows[i]]
        #     padded_lst = torch.cat([foe_dim_i, torch.zeros(cnt_foe - len(foe_dim_i),2).to(self.device)])
        #     foe_dim_n.append(padded_lst)
        # foe_dim_n = torch.stack(foe_dim_n)


        foe_pos_fut = foe_pos.unsqueeze(1).repeat(1,cnt_ego,1,1)
        foe_vel_fut = foe_vel.unsqueeze(1).repeat(1,cnt_ego,1,1)
        ego_pos_fut = ego_pos_n.unsqueeze(2).repeat(1,1, cnt_foe, 1)
        ego_vel_fut = ego_vel_n.unsqueeze(2).repeat(1,1, cnt_foe, 1)

        ego_dim_fut = ego_dim.unsqueeze(1).unsqueeze(2).repeat(self.out_length, cnt_ego, cnt_foe, 1)
        foe_dim_fut = foe_dim.unsqueeze(1).unsqueeze(2).repeat(1,self.out_length, cnt_ego, 1, 1)
        foe_dim_fut = foe_dim_fut.reshape(-1,cnt_ego, cnt_foe,2)

        s_risk_fut, o_risk_fut, ttc_fut = self.cal_risk(ego_pos_fut, ego_vel_fut, ego_dim_fut, foe_pos_fut, foe_vel_fut,
                                               foe_dim_fut)
        # ttc = self.cal_ttc(ego_pos_fut, ego_vel_fut, foe_pos_fut, foe_vel_fut,ego_dim_fut, foe_dim_fut)
        s_risk_fut = s_risk_fut.reshape(batch_size, self.out_length, cnt_ego).contiguous().mean(dim=2)
        o_risk_fut = o_risk_fut.reshape(batch_size, self.out_length, cnt_ego).contiguous().mean(dim=2)
        # ttc_fut = ttc.reshape(batch_size, self.out_length, cnt_ego).contiguous().mean(dim=2)
        return s_risk_fut, o_risk_fut
    
    def get_endpoint_risk(self, ego_pos,ego_vel, foe_future, grid_feature):
        ego_pos = ego_pos * 0.3048# + ego_current1
        ego_vel = ego_vel *0.3048
        foe_pos = foe_future[:, :, 0:2]
        foe_vel = foe_future[:, :, 2:4]
        ego_dim = grid_feature[:,0,0,5:7] * 0.3048
        foe_dim = grid_feature[:,1:,0,5:7] * 0.3048

        cnt = foe_pos.shape[1]
        batch_size = ego_pos.shape[0]
        foe_pos_fut = foe_pos.unsqueeze(1).repeat(1,ego_pos.shape[1],1,1)
        foe_vel_fut = foe_vel.unsqueeze(1).repeat(1,ego_vel.shape[1],1,1)
        ego_pos_fut = ego_pos.unsqueeze(2).repeat(1,1, cnt, 1)
        ego_vel_fut = ego_vel.unsqueeze(2).repeat(1,1, cnt, 1)

        ego_dim_fut = ego_dim.unsqueeze(1).unsqueeze(2).repeat(1,ego_pos.shape[1], cnt, 1)
        foe_dim_fut = foe_dim.unsqueeze(1).repeat(1,ego_pos.shape[1], 1, 1)

        s_risk_fut, o_risk_fut, ttc_fut = self.cal_risk(ego_pos_fut, ego_vel_fut, ego_dim_fut, foe_pos_fut, foe_vel_fut,
                                               foe_dim_fut)
        s_risk = s_risk_fut.sum().cpu().numpy()/batch_size
        o_risk = o_risk_fut.sum().cpu().numpy()/batch_size
        ttc = []#ttc_fut.sum(dim=2)
        return s_risk, o_risk, ttc

    def cal_risk_xy(self, ego_pos, ego_vel, ego_dim, foe_pos, foe_vel, foe_dim, ego_psi):
        dx_ori = ego_pos[:, :, :, :, 0] - foe_pos[:, :, :, :, 0] #大
        dy_ori = ego_pos[:, :, :, :, 1] - foe_pos[:, :, :, :, 1] #小
        if self.dataset == "shrp2":
            dx = dx_ori*torch.cos(ego_psi)+dy_ori*torch.sin(ego_psi)
            dy = -dx_ori*torch.sin(ego_psi)+dy_ori*torch.cos(ego_psi)
            foe_dim[:, :, :, :, 0] = 2
        else:
            dx = dx_ori
            dy = dy_ori
        # define an epsilon to prevent 0 distance
        epsilon = 1e-2
        # get dx and dy WITH vehicle shapes
        dx = torch.clamp(torch.abs(dx) - 0.5 * (ego_dim[:, :, :, :, 0] + foe_dim[:, :, :, :, 0]), 0) * dx / (abs(dx) + epsilon)
        dy = torch.clamp(torch.abs(dy) - 0.5 * (ego_dim[:, :, :, :, 1] + foe_dim[:, :, :, :, 1]), 0) * dy / (abs(dy) + epsilon)

        norm = torch.sqrt(dx**2 + dy**2 + 1e-8)
        cos_theta = torch.abs(dx / norm)
        sin_theta = torch.abs(dy / norm)*5
        order_x = (dx > 0).to(torch.int)
        order_y = (dy > 0).to(torch.int)

        if self.dataset == "shrp2":
            dVx = -(torch.cos(ego_psi)*(ego_vel[:, :, :, :, 0] - foe_vel[:, :, :, :, 0])+
                    torch.sin(ego_psi)*(ego_vel[:, :, :, :, 1] - foe_vel[:, :, :, :, 1])) * dx / (abs(dx) + epsilon)
            dVy = -(torch.sin(ego_psi)*foe_vel[:, :, :, :, 0]-torch.cos(ego_psi)*foe_vel[:, :, :, :, 1]) * dy / (abs(dy) + epsilon)
        else:
            dVx = -((ego_vel[:, :, :, :, 0]) - (foe_vel[:, :, :, :, 0])) * dx / (abs(dx) + epsilon)
            dVy = -(ego_vel[:, :, :, :, 1] - foe_vel[:, :, :, :, 1]) * dy / (abs(dy) + epsilon)


        vj = torch.abs(ego_vel[:, :, :, :, 0])
        if self.dataset == "highd":
            # gx = vj *(1+self.risk_para[0]) + self.risk_para[1]  #1.5 2
            # gy = vj *(0.03+self.risk_para[2])+ self.risk_para[3] #1.5 1
            # bx = vj *(0.04+self.risk_para[4]) + self.risk_para[5]+2#2
            # by = self.risk_para[6] + 5.5
            gx = vj  #1.5+2
            gy = vj * 0.03 #+1
            bx = vj * 0.04 + 2
            by = 5.5
            # gx = torch.clamp(gx, min=1)
            # gy = torch.clamp(gy, min=1)
            # bx = torch.clamp(bx, min=1)
            # by = torch.clamp(by, min=1)
        elif self.dataset == "shrp2":
            # gx = torch.abs(ego_vel[:, :, :, :, 0]) *1.5 + 2 #1.5 2
            # gy = torch.abs(ego_vel[:, :, :, :, 1]) *1.5 + 1 #1.5 1
            # bx = torch.abs(ego_vel[:, :, :, :, 0]) * 0.04 + 2#2
            # by = 5.5
            gx = torch.abs(ego_vel[:, :, :, :, 0]) *(1.5+self.risk_para[0]) + self.risk_para[1]+2  #1.5 2
            gy = torch.abs(ego_vel[:, :, :, :, 1]) *(1.5+self.risk_para[2])+ self.risk_para[3]+1 #1.5 1
            bx = torch.abs(ego_vel[:, :, :, :, 0]) *(0.04+self.risk_para[4]) + self.risk_para[5]+2#2
            by = self.risk_para[6] + 5.5
            gx = torch.clamp(gx, min=1)
            gy = torch.clamp(gy, min=1)
            bx = torch.clamp(bx, min=1)
            by = torch.clamp(by, min=1)


        # s_para = self.s_risk_para(ego_vel)
        # s_para = s_para + 1
        # gx = s_para[:, :, :, 0]
        # gy = s_para[:, :, :, 1]

        # get s_risks
        s_risk = torch.exp(-torch.abs(dx / gx) ** bx - torch.abs(dy / gy) ** by) * (foe_pos[:, :, :, :, 0] != 0).float()
        # s_risk = torch.exp(-2*(torch.abs(dx / gx) ** bx + torch.abs(dy / gy) ** by - 1)) * (foe_pos[:,:,:,:,0] != 0).float()
        s_risk_x = s_risk * cos_theta
        s_risk_y = s_risk * sin_theta

        veh_srisk_x_right = s_risk_x * order_x
        veh_srisk_x_left = s_risk_x * (1 - order_x)
        veh_srisk_x_right_sum = veh_srisk_x_right.sum(dim=2)  
        veh_srisk_x_left_sum = veh_srisk_x_left.sum(dim=2)  

        veh_srisk_y_up = s_risk_y * order_y
        veh_srisk_y_down = s_risk_y * (1 - order_y)
        veh_srisk_y_up_sum = veh_srisk_y_up.sum(dim=2)
        veh_srisk_y_down_sum = veh_srisk_y_down.sum(dim=2)
        
        if self.dataset == "highd":
            _d = 1#3 risky 5  highd 3/1
            _t = 2#5 risky 7.5 highd 5/2
            _b1 = 5#10  5
            _b2 = 3#2  3
        elif self.dataset == "shrp2":
            # _d = 5  
            # _t = 7.5
            # _b1 = 5#10  5
            # _b2 = 3#2  3
            _d = self.risk_para[7]+5#3.2
            _t = self.risk_para[8]+7.5#4
            _b1 = self.risk_para[9]+5
            _b2 = self.risk_para[10]+3
            _d = torch.clamp(_d, min=1)
            _t = torch.clamp(_t, min=1)
            _b1 = torch.clamp(_b1, min=1)
            _b2 = torch.clamp(_b2, min=1)
        alpha = 1
        beta = 1

        # print('gx ',1.5+self.risk_para[0].item(),' ',self.risk_para[1].item()+2)
        # print('gy ',1.5+self.risk_para[2].item(),' ',self.risk_para[3].item()+1)
        # print('bx ',0.04+self.risk_para[4].item()/20,' ',self.risk_para[5].item()+2)
        # print('by ',by.item())
        # print('alpha1 ',alpha1.item(),' alpha2 ',alpha2.item(),' bias ',bias.item())
        # print('_d',_d.item(),'_t',_t.item(),'_b1',_b1.item(),'_b2',_b2.item())
        # print('alpha ',alpha.item(),' beta ',beta.item(),' bias ',bias1.item())
        # define an epsilon to prevent 0 distance
        epsilon_v = 1e-4
        # define an epsilon to set negative times to a large number
        epsilon_t = 1e-4
        # get o_risks
        
        # v = torch.sqrt(dVx**2 + dVy**2)
        # v_mask = (v != 0) & (dx ** 2 + dy ** 2 > 0)
        # v = torch.clamp(v, epsilon_v)
        # dm2 = (dVx * dy - dVy * dx) ** 2 / (v**2)
        # td = -(dVx * dx + dVy * dy) / (v**2)
        # td = 1 / torch.clamp(1 / (td), min=epsilon_t)

        v = torch.sqrt(dVx**2 + dVy**2)
        v_mask = (v != 0) & (dx ** 2 + dy ** 2 > 0)
        dm2 = (dVx * dy - dVy * dx) ** 2 / (dVx**2 + dVy**2 + epsilon_v)
        td = -(dVx * dx + dVy * dy) / (dVx**2 + dVy**2 + epsilon_v)
        td = 1 / torch.clamp(1 / (td+epsilon_t), min=epsilon_t)

        o_risk = torch.exp(-alpha * (dm2 / _d) ** _b1 ) * torch.exp(-beta * (td / _t) ** _b2) * v_mask
        # o_risk = torch.exp(-alpha * (dm2 / _d) ** _b1 + 1) * torch.exp(-beta * (td / _t) ** _b2 + 1) * v_mask
        o_risk = o_risk * (foe_pos[:, :, :, :, 0] != 0).float()


        o_risk_x = o_risk * cos_theta
        o_risk_y = o_risk * sin_theta

        veh_orisk_x_right = o_risk_x * order_x
        veh_orisk_x_left = o_risk_x * (1 - order_x)
        veh_orisk_x_right_sum = veh_orisk_x_right.sum(dim=2)
        veh_orisk_x_left_sum = veh_orisk_x_left.sum(dim=2)

        veh_orisk_y_up = o_risk_y * order_y
        veh_orisk_y_down = o_risk_y * (1 - order_y)
        veh_orisk_y_up_sum = veh_orisk_y_up.sum(dim=2)
        veh_orisk_y_down_sum = veh_orisk_y_down.sum(dim=2)

        # ttc = self.cal_ttc(ego_pos, ego_vel, foe_pos, foe_vel, ego_dim, foe_dim)

        # ttc_x = ttc * cos_theta
        # ttc_y = ttc * sin_theta

        # veh_ttc_x_right = ttc_x * order_x
        # veh_ttc_x_left = ttc_x * (1 - order_x)
        # veh_ttc_x_right_sum = veh_ttc_x_right.sum(dim=2)
        # veh_ttc_x_left_sum = veh_ttc_x_left.sum(dim=2)

        # veh_ttc_y_up = ttc_y * order_y
        # veh_ttc_y_down = ttc_y * (1 - order_y)
        # veh_ttc_y_up_sum = veh_ttc_y_up.sum(dim=2)
        # veh_ttc_y_down_sum = veh_ttc_y_down.sum(dim=2)

        s_risk_sum = torch.stack([veh_srisk_x_right_sum, veh_srisk_x_left_sum, veh_srisk_y_up_sum, veh_srisk_y_down_sum], dim=3)
        o_risk_sum = torch.stack([veh_orisk_x_right_sum, veh_orisk_x_left_sum, veh_orisk_y_up_sum, veh_orisk_y_down_sum], dim=3)
        ttc_sum = []#torch.stack([veh_ttc_x_right_sum, veh_ttc_x_left_sum, veh_ttc_y_up_sum, veh_ttc_y_down_sum], dim=2)

        return s_risk_sum, o_risk_sum, ttc_sum
    
    def cal_risk_grid(self, ego_pos, ego_vel, ego_dim, foe_pos, foe_vel, foe_dim, ego_psi):
        dx_ori = ego_pos[:, :, :, :, 0] - foe_pos[:, :, :, :, 0] #大
        dy_ori = ego_pos[:, :, :, :, 1] - foe_pos[:, :, :, :, 1] #小
        if self.dataset == "shrp2":
            dx = dx_ori*torch.cos(ego_psi)+dy_ori*torch.sin(ego_psi)
            dy = -dx_ori*torch.sin(ego_psi)+dy_ori*torch.cos(ego_psi)
            foe_dim[:, :, :, :, 0] = 2
        else:
            dx = dx_ori
            dy = dy_ori
        # define an epsilon to prevent 0 distance
        epsilon = 1e-2
        # get dx and dy WITH vehicle shapes
        dx = torch.clamp(torch.abs(dx) - 0.5 * (ego_dim[:, :, :, :, 0] + foe_dim[:, :, :, :, 0]), 0) * dx / (abs(dx) + epsilon)
        dy = torch.clamp(torch.abs(dy) - 0.5 * (ego_dim[:, :, :, :, 1] + foe_dim[:, :, :, :, 1]), 0) * dy / (abs(dy) + epsilon)


        if self.dataset == "shrp2":
            dVx = -(torch.cos(ego_psi)*(ego_vel[:, :, :, :, 0] - foe_vel[:, :, :, :, 0])+
                    torch.sin(ego_psi)*(ego_vel[:, :, :, :, 1] - foe_vel[:, :, :, :, 1])) * dx / (abs(dx) + epsilon)
            dVy = -(torch.sin(ego_psi)*foe_vel[:, :, :, :, 0]-torch.cos(ego_psi)*foe_vel[:, :, :, :, 1]) * dy / (abs(dy) + epsilon)
        else:
            dVx = -(torch.abs(ego_vel[:, :, :, :, 0]) - torch.abs(foe_vel[:, :, :, :, 0])) * dx / (abs(dx) + epsilon)
            dVy = -(ego_vel[:, :, :, :, 1] - foe_vel[:, :, :, :, 1]) * dy / (abs(dy) + epsilon)


        vj = torch.abs(ego_vel[:, :, :, :, 0])
        if self.dataset == "highd":
            gx = vj  #1.5+2
            gy = vj * 0.03 #+1
            bx = vj * 0.04 + 2
            by = 5.5
        elif self.dataset == "shrp2":
            gx = torch.abs(ego_vel[:, :, :, :, 0]) *1.5 + 2 #1.5 2
            gy = torch.abs(ego_vel[:, :, :, :, 1]) *1.5 + 1 #1.5 1
            bx = torch.abs(ego_vel[:, :, :, :, 0]) * 0.04 + 2#2
            by = 5.5
            # gx = torch.abs(ego_vel[:, :, :, :, 0]) *(1.5+self.risk_para[0]*10) + self.risk_para[1]*10+2  #1.5 2
            # gy = torch.abs(ego_vel[:, :, :, :, 1]) *(1.5+self.risk_para[2]*10)+ self.risk_para[3]*10+1 #1.5 1
            # bx = torch.abs(ego_vel[:, :, :, :, 0]) *(0.04+self.risk_para[4]*10) + self.risk_para[5]*10+2#2
            # by = self.risk_para[6]*10 + 3
            # gx = torch.clamp(gx, min=1)
            # gy = torch.clamp(gy, min=1)
            # bx = torch.clamp(bx, min=1)
            # by = torch.clamp(by, min=1)



        # get s_risks
        # s_risk = torch.exp(-torch.abs(dx / gx) ** bx - torch.abs(dy / gy) ** by) * (foe_pos[:, :, :, :, 0] != 0).float()
        s_risk = torch.exp(-2*(torch.abs(dx / gx) ** bx + torch.abs(dy / gy) ** by - 1)) * (foe_pos[:,:,:, :,0] != 0).float()

        
        _d = 5#3 risky 5  highd 1
        _t = 7.5#5 risky 7.5 highd 2
        _b1 = 5#10  5
        _b2 = 3#2  3
        # _d = self.risk_para[7]*10+5
        # _t = self.risk_para[8]*10+7.5
        # _b1 = self.risk_para[9]*10+5
        # _b2 = self.risk_para[10]*10+3
        # _d = torch.clamp(_d, min=1)
        # _t = torch.clamp(_t, min=1)
        # _b1 = torch.clamp(_b1, min=1)
        # _b2 = torch.clamp(_b2, min=1)
        alpha = 2 #1
        beta = 2 #1
        # define an epsilon to prevent 0 distance
        epsilon_v = 1e-4
        # define an epsilon to set negative times to a large number
        epsilon_t = 1e-4
        # get o_risks
        v = torch.sqrt(dVx**2 + dVy**2)
        v_mask = (v != 0) & (dx ** 2 + dy ** 2 > 0)
        # v = torch.clamp(v, epsilon_v)
        dm2 = (dVx * dy - dVy * dx) ** 2 / (dVx**2 + dVy**2 + epsilon_v)
        td = -(dVx * dx + dVy * dy) / (dVx**2 + dVy**2 + epsilon_v)
        td = 1 / torch.clamp(1 / (td+epsilon_t), min=epsilon_t)
        # td = torch.where(td > 1/epsilon_t, 1/epsilon_t, td)
        # td = torch.where(td < 0, 1 / epsilon_t, td)
        # o_risk = torch.exp(-alpha * (dm2 / _d) ** _b1 ) * torch.exp(-beta * (td / _t) ** _b2 ) * v_mask
        o_risk = torch.exp(-alpha * (dm2 / _d) ** _b1 + 1) * torch.exp(-beta * (td / _t) ** _b2 + 1) * v_mask
        o_risk = o_risk * (foe_pos[:, :, :, :, 0] != 0).float()


        # ttc = self.cal_ttc(ego_pos, ego_vel, foe_pos, foe_vel, ego_dim, foe_dim)

        s_risk_sum = s_risk.unsqueeze(4)
        o_risk_sum = torch.cat((o_risk.unsqueeze(4), foe_pos, foe_vel), dim=-1)
        ttc_sum = []#torch.stack([veh_ttc_x_right_sum, veh_ttc_x_left_sum, veh_ttc_y_up_sum, veh_ttc_y_down_sum], dim=2)

        return s_risk_sum, o_risk_sum, ttc_sum
    
    def cal_ttc(self, pos1, vel1, pos2, vel2, radius1, radius2):
        # Relative position and velocity
        batch_size, num_ego, num_foe, _ = pos1.shape
        pos1 = pos1.reshape(-1, 2)
        vel1 = vel1.reshape(-1, 2)
        pos2 = pos2.reshape(-1, 2)
        vel2 = vel2.reshape(-1, 2)
        radius1 = radius1.reshape(-1, 2)
        radius2 = radius2.reshape(-1, 2)
        # rel_pos = pos2 - pos1
        # rel_vel = vel2 - vel1
        ego_psi = torch.atan2(vel1[:,1], vel1[:,0])
        dx_ori = pos2[:,0] - pos1[:,0]
        dy_ori = pos2[:,1] - pos1[:,1]
        dvx_ori = vel2[:,0] - vel1[:,0]
        dvy_ori = vel2[:,1] - vel1[:,1]

        dx = dx_ori*torch.cos(ego_psi)+dy_ori*torch.sin(ego_psi)
        dy = -dx_ori*torch.sin(ego_psi)+dy_ori*torch.cos(ego_psi)
        rel_pos = torch.stack((dx, dy), dim=1)

        dVx = dvx_ori*torch.cos(ego_psi)+dvy_ori*torch.sin(ego_psi)
        dVy = -dvx_ori*torch.sin(ego_psi)+dvy_ori*torch.cos(ego_psi)
        rel_vel = torch.stack((dVx, dVy), dim=1)

        # Combined radius
        r = (radius1[:,1] + radius2[:,1])/2

        # Coefficients for quadratic equation: a*t^2 + b*t + c = 0
        a = torch.sum(rel_vel * rel_vel, dim=1)  # dot product of relative velocity
        b = 2 * torch.sum(rel_pos * rel_vel, dim=1)  # 2 * dot product of relative position and velocity
        c = torch.sum(rel_pos * rel_pos, dim=1) - r ** 2  # adjusted for radii

        # Initialize TTC array with infinity
        ttc = torch.full(a.shape, float('inf')).to(pos1.device)

        # Avoid division by zero where relative velocity is zero
        non_zero_a = torch.abs(a) > 1e-10

        # Calculate discriminant
        discriminant = b[non_zero_a] ** 2 - 4 * a[non_zero_a] * c[non_zero_a]

        # Only consider cases with real roots (discriminant >= 0)
        valid = discriminant >= 0
        valid_indices = torch.where(non_zero_a)[0][valid]

        # Solve for positive roots
        sqrt_disc = torch.sqrt(discriminant[valid])
        a_valid = a[valid_indices]
        b_valid = b[valid_indices]

        # Calculate possible collision times
        t1 = (-b_valid + sqrt_disc) / (2 * a_valid)
        t2 = (-b_valid - sqrt_disc) / (2 * a_valid)

        # # Select the smallest positive time
        # for i, idx in enumerate(valid_indices):
        #     times = [t for t in [t1[i], t2[i]] if t > 0]
        #     if times:
        #         ttc[idx] = min(times)

        valid_times = torch.stack([t1, t2], dim=-1)
        valid_times = valid_times.min(dim=-1)[0]
        valid_times[valid_times <= 0] = float('inf')  
        valid_times = valid_times.float()
        ttc[valid_indices] = valid_times
        
        ttc = ttc.reshape(batch_size, num_ego, num_foe)
        ttc_min, _ = ttc.min(dim=2)
        ttc_min = 10/(ttc_min+1e-6)
        return ttc_min



class Criterion(nn.Module):
    def __init__(self, args):
        super(Criterion, self).__init__()
        self.entropy_weight = args['entropy_weight']
        self.kl_weight = args['kl_weight']
        self.use_FDEADE_aux_loss = args['use_FDEADE_aux_loss']

    def forward(self, out, gt, mask):

        return self.nll_loss_multimodes(out, gt, mask)

    def get_BVG_distributions(self, pred):
        B = pred.size(0)
        T = pred.size(1)
        mu_x = pred[:, :, 0].unsqueeze(2)
        mu_y = pred[:, :, 1].unsqueeze(2)
        sigma_x = pred[:, :, 2]
        sigma_y = pred[:, :, 3]
        rho = pred[:, :, 4]

        # Create the base covariance matrix for a single element
        cov = torch.stack([
            torch.stack([sigma_x ** 2, rho * sigma_x * sigma_y], dim=-1),
            torch.stack([rho * sigma_x * sigma_y, sigma_y ** 2], dim=-1)
        ], dim=-2)

        # Expand this base matrix to match the desired shape
        biv_gauss_dist = MultivariateNormal(loc=torch.cat((mu_x, mu_y), dim=-1), covariance_matrix=cov,validate_args=False)
        return biv_gauss_dist

    def get_Laplace_dist(self, pred):
        return Laplace(pred[:, :, :2], pred[:, :, 2:4],validate_args=False)

    def nll_pytorch_dist(self, pred, data, mask, rtn_loss=True):
        # biv_gauss_dist = get_BVG_distributions(pred)
        biv_gauss_dist = self.get_Laplace_dist(pred)
        num_active_per_timestep = mask.sum()
        data_reshaped = data
        if rtn_loss:
            # return (-biv_gauss_dist.log_prob(data)).sum(1)  # Gauss
            return ((-biv_gauss_dist.log_prob(data_reshaped)).sum(-1) * mask).sum(1)  # Laplace
        else:
            # return (-biv_gauss_dist.log_prob(data)).sum(-1)  # Gauss
            # need to multiply by masks
            # return (-biv_gauss_dist.log_prob(data_reshaped)).sum(dim=(1, 2))  # Laplace
            return ((-biv_gauss_dist.log_prob(data_reshaped)).sum(dim=2) * mask).sum(1)  # Laplace

    def nll_loss_multimodes(self, output, data, mask):
        """NLL loss multimodes for training. MFP Loss function
        Args:
          pred: [K, T, B, 5]
          data: [B, T, 5]
          modes_pred: [B, K], prior prob over modes
          noise is optional
        """
        modes_pred = output['predicted_probability']
        pred = output['predicted_trajectory'].permute(1, 2, 0, 3) # K,T,B,5

        modes = len(pred)
        nSteps, batch_sz, dim = pred[0].shape

        log_lik_list = []
        with torch.no_grad():
            for kk in range(modes):
                nll = self.nll_pytorch_dist(pred[kk].transpose(0, 1), data, mask, rtn_loss=False)
                log_lik_list.append(-nll.unsqueeze(1))  # Add a new dimension to concatenate later

        # Concatenate the list to form the log_lik tensor
            log_lik = torch.cat(log_lik_list, dim=1)

            priors = modes_pred
            log_priors = torch.log(priors)
            log_posterior_unnorm = log_lik + log_priors

            # Compute logsumexp for normalization, ensuring no in-place operations
            logsumexp = torch.logsumexp(log_posterior_unnorm, dim=-1, keepdim=True)
            log_posterior = log_posterior_unnorm - logsumexp

            # Compute the posterior probabilities without in-place operations
            post_pr = torch.exp(log_posterior)
            # Ensure post_pr is a tensor on the correct device
            post_pr = post_pr.to(data.device)

        # Compute loss.
        loss = 0.0
        for kk in range(modes):
            nll_k = self.nll_pytorch_dist(pred[kk].transpose(0, 1), data, mask, rtn_loss=True) * post_pr[:, kk]
            loss += nll_k.mean()

        # Adding entropy loss term to ensure that individual predictions do not try to cover multiple modes.
        entropy_vals = []
        for kk in range(modes):
            entropy_vals.append(self.get_BVG_distributions(pred[kk]).entropy())
        entropy_vals = torch.stack(entropy_vals).permute(2, 0, 1)
        entropy_loss = torch.mean((entropy_vals).sum(2).max(1)[0])
        loss += self.entropy_weight * entropy_loss

        # KL divergence between the prior and the posterior distributions.
        # kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')  # type: ignore
        # kl_loss = self.kl_weight * kl_loss_fn(torch.log(modes_pred), post_pr)

        # compute ADE/FDE loss - L2 norms with between best predictions and GT.
        if self.use_FDEADE_aux_loss:
            adefde_loss = self.l2_loss_fde(pred, data, mask)
        else:
            adefde_loss = torch.tensor(0.0).to(data.device)

        # post_entropy
        final_loss = loss  + adefde_loss

        return final_loss

    def l2_loss_fde(self, pred, data, mask):

        fde_loss = (torch.norm((pred[:, -1, :, :2].transpose(0, 1) - data[:, -1, :].unsqueeze(1)), 2, dim=-1) * mask[:,
                                                                                                                 -1:])
        ade_loss = (torch.norm((pred[:, :, :, :2].transpose(1, 2) - data[:, :, :].unsqueeze(0)), 2,
                               dim=-1) * mask.unsqueeze(0)).mean(dim=2).transpose(0, 1)
        loss, min_inds = (fde_loss + ade_loss).min(dim=1)
        return 100.0 * loss.mean()


