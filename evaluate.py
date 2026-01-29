from __future__ import print_function

import numpy as np
import torch
from config import *

import loader_risky_time as lo_shrp
import loader_highD as lo_highd
from torch.utils.data import DataLoader
import pandas as pd

import matplotlib.pyplot as plt
import os
import time
from torch.utils.tensorboard import SummaryWriter
import random
from torch.utils.data import Subset
from gnn_model_risk_large import *
from FoePredict import FoePredict_risky_time, FoePredict_highd_time
from risk_decoder_autobot_v1_time import Risk_Decoder_auto_time
import matplotlib.cm as cm
import pickle
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



class Evaluate():

    def __init__(self):
        self.op = 0
        self.drawImg = False
        self.scale = 0.3048
        self.prop = 1


    def maskedMSETest(self, y_pred, y_gt, mask):
        g_out = y_pred.permute(1, 0, 2)
        muX = g_out[:, :, 0]
        muY = g_out[:, :, 1]
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        out = t.pow(x - muX, 2) + t.pow(y - muY, 2)
        out = out * mask

        lossVal = t.sum(out)
        counts = t.sum(mask)
        loss = t.sum(out) / t.sum(mask)
        return lossVal, counts, loss


    def MSELoss_goal(self, foe_pred, foe_gt, adj, mask):
        out = t.pow(foe_pred - foe_gt, 2) * mask[:, None, None, None]  #[:, 1:, :]  
        return (t.sum(out)) / t.sum(adj)
    
    

    def MSELoss_goal_simple(self, foe_pred, foe_gt, adj):
        out = t.pow(foe_pred - foe_gt, 2)
        return t.sum(out) / t.sum(adj)


    def maskedMSETestReal(self, y_pred, y_gt, mask):
        y_gt = y_gt.permute(1, 0, 2)
        mask = mask.permute(1, 0).unsqueeze(2).repeat(1, 1, 2)
        y_pred = y_pred * mask
        y_gt = y_gt * mask
        acc = t.pow(y_gt - y_pred, 2)
        return t.sum(acc, dim=(1,2)), t.sum(mask[:, :, 0], dim=1), t.sum(acc, dim=(0,2))
    

    def ADE_FDETestReal(self, y_pred, y_gt, mask):
        y_gt = y_gt.permute(1, 0, 2)
        mask = mask.permute(1, 0).unsqueeze(2).repeat(1, 1, 2)
        y_pred = y_pred * mask
        y_gt = y_gt * mask

        displacement = t.norm(y_pred - y_gt, dim=2)
        ade = displacement.sum()

        fde = displacement[-1,:].sum()
        return ade, t.sum(mask[:, :, 0]), fde, t.sum(mask[-1, :, 0]), displacement.mean(dim=0), displacement[-1,:]
    


    def NLLLoss_score(self, pred_scores, pred, target, mask, log_std_range=(-1.609, 5.0), rho_limit=0.5):
        batch_size = pred_scores.shape[0]
        distance = (pred[:, :, :, 0:2] - target[:, None, :, :]).norm(dim=-1)
        distance = (distance * mask[:, None, :]).sum(dim=-1)
        nearest_mode_idxs = distance.argmin(dim=-1)
        preds_mode_idxs = pred_scores.argmax(dim=-1)
        nearest_mode_bs_idxs = t.arange(batch_size).type_as(nearest_mode_idxs) 

        maxprob_trajs = pred[nearest_mode_bs_idxs, preds_mode_idxs]
        nearest_trajs = pred[nearest_mode_bs_idxs, nearest_mode_idxs]

        mu_x, mu_y = nearest_trajs[:, :, 0], nearest_trajs[:, :, 1]
        sigma_x, sigma_y = nearest_trajs[:, :, 2], nearest_trajs[:, :, 3]
        rho = nearest_trajs[:, :, 4]
        y_x, y_y = target[:, :, 0], target[:, :, 1]

        dx = y_x - mu_x
        dy = y_y - mu_y
        log_std1 = t.clip(sigma_x, min=log_std_range[0], max=log_std_range[1])
        log_std2 = t.clip(sigma_y, min=log_std_range[0], max=log_std_range[1])
        std1 = t.exp(log_std1)  # (0.2m to 150m)
        std2 = t.exp(log_std2)  # (0.2m to 150m)
        rho = t.clip(rho, min=-rho_limit, max=rho_limit)

        reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * t.log(1 - rho ** 2)  # (batch_size, num_timestamps)
        reg_gmm_exp = (0.5 * 1 / (1 - rho ** 2)) * (
                (dx ** 2) / (std1 ** 2) + (dy ** 2) / (std2 ** 2) - 2 * rho * dx * dy / (
                std1 * std2))  # (batch_size, num_timestamps)

        nll_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * mask).sum(dim=-1)

        # distance = (pred[:, :, :, 0:2] - target[:, None, :, :]).norm(dim=-1)
        # distance = (distance * mask[:, None, :]).sum(dim=-1)
        # nearest_mode_idxs = distance.argmin(dim=-1)
        # correct = (preds_mode_idxs == nearest_mode_idxs).sum().item()
        # accuracy = correct / nearest_mode_idxs.size(0)
        return t.sum(nll_loss) / t.sum(mask), nearest_trajs, maxprob_trajs

    def main(self, name, val, writer,dataset):
        args['path'] = 'checkpoint/' + dataset
        args['dataset'] = dataset  # highd shrp2
        args['ego_goals_path'] = 'data/' + dataset + '/'  ###shrp2 highd
        model_step = 1
        args['train_flag'] = False
        l_path = args['path']
        

        if args['dataset'] == "shrp2":
            foe_predict = FoePredict_risky_time(args)
            generator = Risk_Decoder_auto_time(args=args)
        else:
            foe_predict = FoePredict_highd_time(args)
            generator = Risk_Decoder_auto_time(args=args)
        gdEncoder_all = GDEncoder_all(args)


        generator.load_state_dict(t.load(l_path + '/epoch' + name + '_g.tar')) # , map_location='cuda:1'
        foe_predict.load_state_dict(t.load(l_path + '/epoch' + name + '_f.tar'))
        gdEncoder_all.load_state_dict(t.load(l_path + '/epoch' + name + '_gda.tar'))

        generator = generator.to(device)
        foe_predict = foe_predict.to(device)
        gdEncoder_all = gdEncoder_all.to(device)

        generator.eval()
        foe_predict.eval()
        gdEncoder_all.eval()


        if val:
            if args['dataset'] == "shrp2":
                t2 = lo_shrp.RiskyDataset('shrp2/ValSetRisky_all1.mat')
            else:
                t2 = lo_highd.HighdDataset('highD/','highD/','val')
                args['num_worker'] = 2

            valDataloader = DataLoader(t2, batch_size=args['batch_size'], shuffle=False,
                                       num_workers=args['num_worker'],
                                       collate_fn=t2.collate_fn)
        else:
            if args['dataset'] == "highd":
                t2 = lo_highd.HighdDataset('data/highD/','data/highD/', 'test')
                args['num_worker'] = 1
            else:
                t2 = lo_shrp.RiskyDataset('data/shrp2/test_shrp_mini.mat')
     
            valDataloader = DataLoader(t2, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_worker'],
                                       collate_fn=t2.collate_fn)

        lossVals = t.zeros(args['out_length']).to(device)
        counts = t.zeros(args['out_length']).to(device)
        avg_val_loss = 0
        all_time = 0
        mse_loss_real = t.zeros(args['out_length'])
        count_loss_real = t.zeros(args['out_length'])
        mse_loss_real_maxp = t.zeros(args['out_length'])
        count_loss_real_maxp = t.zeros(args['out_length'])
        nll_loss = 0
        ade_loss = 0
        ade_count = 0
        fde_loss = 0
        fde_count = 0
        goal_loss = 0
        goal_loss_simple = 0
        risks_loss = 0

        hist_record = np.empty((0,16,4))
        fut_record = np.empty((0,25,4))
        fut_pred_record = np.empty((0,6,25,7))
        fut_pred_best_record = np.empty((0,25,7))
        fut_pred_probability = np.empty((0,6))
        dim_record = np.empty((0,2))
        
        neighborhood_hist = []
        neighborhood_dim = []
        neighbor_goal = []
        neighbor_goal_pred = []
        velocity_record = []
        neighbor_fut_record = []
        s_risk_all = [0,0]
        o_risk_all = [0,0]
        risk_acc = np.empty((0,6))
        val_batch_count = len(valDataloader)
        transition_attn = np.empty((0,100,4,5))
        s_risk_fut_record = np.empty((0,))
        o_risk_fut_record = np.empty((0,))
        ego_current_record = np.empty((0,))
        goal_error_record = t.empty((0,10,4)).to(device)
        ds_Ids = []
        veh_Ids = []
        frame_Ids = []
        fut_risks = []
        ade_loss =np.zeros(int(args['out_length']/5))
        fde_loss =np.zeros(int(args['out_length']/5))
        ade_count =np.zeros(int(args['out_length']/5))
        fde_count =np.zeros(int(args['out_length']/5))
        traj_idx=0
        # with open("checkpoint/ds_veh_t.pkl", "rb") as f:
        #     ds_veh_original = pickle.load(f)
        print("begin.................................", name)
        with(t.no_grad()):
            for idx, data in enumerate(valDataloader):
                hist, fut, mask, grid_feature, adj, dsId, vehId, frame_Id, ego_current,v_fut, foe_future, neighbor_fut, maneuver = data
                grid_feature = grid_feature.to(device)
                # hist = hist.to(device)
                mask = mask.to(device)
                fut = fut.to(device)
                adj = adj.to(device)
                foe_future = foe_future.to(device)
                v_fut = v_fut.to(device)

                values_all = gdEncoder_all(grid_feature, adj)
                foe_goal = foe_predict(grid_feature[:, :, :, :], adj[:, :], values_all[:,:,:,:]) ###grid_feature[:, 1:, :, 0:4], adj[:, 1:], values_all[:,1:,:,:]
    
                loss_goal = self.MSELoss_goal(foe_goal[:,:,:,:], foe_future[:, :, 1:,0:4], adj[:, :],mask[:,0])  ###adj[:, 1:]

             

                foe_goal = t.concat((foe_future[:, :, 0:1, 0:4], foe_goal), dim=2)
                fut_pred1 = generator(values_all[:,0,:,:], ego_current, foe_goal[:, :, :, :], grid_feature) ###fvalues_all[:,0,:,:]
              

                # transition_attn = np.concatenate((transition_attn, fut_pred1['transition_probs'].cpu().numpy()), axis=0)
                # s_risk_fut_record = np.concatenate((s_risk_fut_record, grid_feature[:,0,-1,-2].cpu().numpy()), axis=0)
                # o_risk_fut_record = np.concatenate((o_risk_fut_record, grid_feature[:,0,-1,-1].cpu().numpy()), axis=0)
                # ego_current_record = np.concatenate((ego_current_record, fut_pred1['ego_current'].numpy()), axis=0)

                loss_goal_simple = self.MSELoss_goal_simple(foe_future[:, :, 1: ,4:], foe_future[:, :, 1:,0:4], adj[:, :])
    
                loss_nll, fut_pred, max_prob_pred = self.NLLLoss_score(fut_pred1['predicted_probability'], fut_pred1['predicted_trajectory'], fut, mask)
              

                fut_pred = fut_pred.permute(1, 0, 2)
                max_prob_pred = max_prob_pred.permute(1, 0, 2)
                l, c, loss = self.maskedMSETest(fut_pred, fut, mask)
                l1, c1, loss_sample = self.maskedMSETestReal(fut_pred[:,:,0:2], fut, mask)
                l2, c2, loss_sample2 = self.maskedMSETestReal(max_prob_pred[:,:,0:2], fut, mask)
                # ade, ade_c, fde, fde_c = self.ADE_FDETestReal(fut_pred[:,:,0:2], fut, mask)
                for jj in range(5):
                    ade, ade_c, fde, fde_c, ade_i,fde_i = self.ADE_FDETestReal(fut_pred[0:(jj+1)*5,:,0:2], fut[:,0:(jj+1)*5,:], mask[:,0:(jj+1)*5])
                    ade_loss[jj] += ade.item()
                    ade_count[jj] += ade_c.item()
                    fde_loss[jj] += fde.item()
                    fde_count[jj] += fde_c.item()
                
                goal_loss += loss_goal.item()
                goal_loss_simple += loss_goal_simple.item()


                # tmp = np.asarray(maneuver)
                # maneuver = tmp[:,0:1].astype(int)
                # ttc = tmp[:,1:2]
                # risk_ego = tmp[:,2:4]
                # ade_i = ade_i.cpu().numpy()[:, np.newaxis]
                # fde_i = fde_i.cpu().numpy()[:, np.newaxis]
                # risk_acc = np.append(risk_acc, np.concatenate((risk_ego, ttc,ade_i,fde_i, maneuver), axis=1), axis=0) ##

                # tmp = np.asarray(maneuver)
                # maneuver = tmp[:,0].astype(int)
                # ttc = tmp[:,1]
                # risk_ego = tmp[:,2:4]
               
                # for j in range(grid_feature.shape[0]):
                #     # if risk_ego[j,0] > 12 or risk_ego[j,1]>7:#highd算的是risk_ego[j,0] > 7 or risk_ego[j,0]>5
                #     # if (risk_ego[j,0] <0.1 or risk_ego[j,1]<0.1) and adj[j].sum()>3:
                #     if dsId[j] == ds_veh_original['ds_Ids'][traj_idx]:
                #         if vehId[j][0] == ds_veh_original['veh_Ids'][traj_idx]:
                #             if frame_Id[j] == ds_veh_original['frame_Ids'][traj_idx]:
                #                 loss_sample1 = loss_sample[j].cpu().numpy()
                #                 lane = grid_feature[j, 0, :, 4].cpu().numpy() #ngsim ? highd 4
                #                 # fut_traj_lane = fut.cpu().numpy()*0.3048
                #                 # lane_change = np.array(abs(fut_traj_lane[:,-1,0]-fut_traj_lane[:,0,0])>3)[:, np.newaxis]

                #                 # ngsim 10:12  highd 11:13
                #                 risk_acc = np.append(risk_acc, np.array([[risk_ego[j,0], risk_ego[j,1],ttc[j],loss_sample1,loss_sample2[j].cpu().numpy(), maneuver[j]]]), axis=0) ##
                #                 ds_Ids=np.append(ds_Ids,dsId[j]) ##
                #                 veh_Ids=np.append(veh_Ids,vehId[j][0]) ##
                #                 frame_Ids=np.append(frame_Ids, frame_Id[j]) ##
                #                 fut_pred1['predicted_trajectory'][j,:,:,0:4]=fut_pred1['predicted_trajectory'][j,:,:,0:4]*0.3048 ##
                #                 fut_pred1['predicted_trajectory'][j,:,:,5:7]=fut_pred1['predicted_trajectory'][j,:,:,5:7]*0.3048 ##
                #                 fut_pred_record = np.concatenate((fut_pred_record, fut_pred1['predicted_trajectory'][j:(j+1)].cpu().numpy()), axis=0) ##
                #                 fut_pred_probability = np.concatenate((fut_pred_probability, fut_pred1['predicted_probability'][j:(j+1)].cpu().numpy()), axis=0) ##
                #                 fut_record = np.concatenate((fut_record, np.concatenate((fut[j:(j+1)].cpu().numpy()*0.3048, v_fut[j:(j+1)].cpu().numpy()*0.3048), axis=2)), axis=0) ##
                #                 hist_record = np.concatenate((hist_record, grid_feature[j:(j+1), 0, :, 0:4].cpu().numpy()*0.3048), axis=0) ##
                #                 dim_record = np.concatenate((dim_record, (grid_feature[j:(j+1), 0, 0, 5:7].cpu().numpy())*0.3048), axis=0) ##
                #                 neighbor_fut_record.extend(neighbor_fut[j:(j+1)]) ##

                #                 fut_pred_best = fut_pred[:,j:(j+1)].permute(1, 0, 2).cpu().numpy()
                #                 fut_pred_best[0,:,0:4] = fut_pred_best[0,:,0:4]*0.3048
                #                 fut_pred_best[0,:,5:7] = fut_pred_best[0,:,5:7]*0.3048
                #                 fut_pred_best_record = np.concatenate((fut_pred_best_record, fut_pred_best), axis=0) ##
                            
                #                 nrbr_hist = []
                #                 nrbr_dim = []
                #                 for i in range(1,grid_feature.shape[1]):
                #                     if adj[j,i] == 1:
                #                         nrbr = grid_feature[j, i, :, 0:4].cpu().numpy()
                #                         nrbr[:, 0:2] = (nrbr[:, 0:2]+grid_feature[j, i, :, 9:11].cpu().numpy())*0.3048 #ngsim 8:10 highd 9:11
                #                         # nrbr_goal_i = foe_future[j, i-1, 0:4].cpu().numpy()
                #                         # nrbr_goal_pred_i = foe_goal[j, i-1, :].cpu().numpy()
                #                         nrbr_hist.append(nrbr)
                #                         # nrbr_goal.append(nrbr_goal_i)
                #                         # nrbr_goal_pred.append(nrbr_goal_pred_i)
                #                         nrbr_dim.append(grid_feature[j, i, 0, 5:7].cpu().numpy()*0.3048)
                #                         # velocity_record_i.append(grid_feature[j, i, -1, 2:4].cpu().numpy()*0.3048)
                #                 nrbr_hist = np.array(nrbr_hist)
                #                 nrbr_dim = np.array(nrbr_dim)           
                #                 neighborhood_hist.append(nrbr_hist) ##   
                #                 neighborhood_dim.append(nrbr_dim) ##
                #                 traj_idx+=1




# risknet_plot_data = {}
# risknet_plot_data['fut_pred_record'] = fut_pred_record
# risknet_plot_data['fut_pred_probability'] = fut_pred_probability
# risknet_plot_data['fut_record'] = fut_record
# risknet_plot_data['hist_record'] = hist_record
# risknet_plot_data['neighborhood_hist'] = neighborhood_hist
# risknet_plot_data['risk_acc'] = risk_acc
# risknet_plot_data['dim_record'] = dim_record
# risknet_plot_data['neighborhood_dim'] = neighborhood_dim
# risknet_plot_data['fut_pred_best_record'] = fut_pred_best_record
# risknet_plot_data['neighbor_fut_record'] = neighbor_fut_record
# import pickle
# with open('checkpoint/highd_fix5s_noscale.pkl', 'wb') as f:
#     pickle.dump(risknet_plot_data, f)
# ds_veh_t = {}
# ds_veh_t['ds_Ids'] = ds_Ids
# ds_veh_t['veh_Ids'] = veh_Ids
# ds_veh_t['frame_Ids'] = frame_Ids
# with open('checkpoint/shrp_risk_noscale/ds_veh_t_shrp_risk.pkl', 'wb') as f:
#     pickle.dump(ds_veh_t, f)


                avg_val_loss += loss.item()
                mse_loss_real += l1.cpu()
                count_loss_real += c1.cpu()
                mse_loss_real_maxp += l2.cpu()
                count_loss_real_maxp += c2.cpu()
                nll_loss += loss_nll.item()
                if idx == int(val_batch_count / 4) * model_step:
                    print('process:', model_step / 4)
                    model_step += 1
            if args['val_use_mse']:
                print('valmse:', avg_val_loss / val_batch_count)
                print('goal mse:', goal_loss / val_batch_count)
                print('goal simple mse:', goal_loss_simple / val_batch_count)
                # print('risk mse:', risks_loss / val_batch_count)
                if writer is not False:
                    writer.add_scalar('valmse', avg_val_loss / val_batch_count, int(name))
                print('rmse_real in meters:', t.sum(mse_loss_real).item(), t.sum(count_loss_real).item(),
                      t.pow(t.sum(mse_loss_real) / t.sum(count_loss_real),0.5).item() * 0.3048)
                print(t.pow(mse_loss_real / count_loss_real, 0.5) * 0.3048)  # Calculate RMSE and convert from feet to meters
                print('max prob rmse_real in meters:', t.sum(mse_loss_real_maxp).item(), t.sum(count_loss_real_maxp).item(),
                      t.pow(t.sum(mse_loss_real_maxp) / t.sum(count_loss_real_maxp),0.5).item() * 0.3048)
                print(t.pow(mse_loss_real_maxp / count_loss_real_maxp, 0.5) * 0.3048) 
                print('ade in meters: ', ade_loss[0] / ade_count[0] * 0.3048, ade_loss[1] / ade_count[1] * 0.3048, ade_loss[2] / ade_count[2] * 0.3048, ade_loss[3] / ade_count[3] * 0.3048, ade_loss[4] / ade_count[4] * 0.3048)
                print('fde in meters: ', fde_loss[0] / fde_count[0] * 0.3048, fde_loss[1] / fde_count[1] * 0.3048, fde_loss[2] / fde_count[2] * 0.3048, fde_loss[3] / fde_count[3] * 0.3048, fde_loss[4] / fde_count[4] * 0.3048)
                print('valnll:', nll_loss / val_batch_count)
              

                # risk_acc_cluster = []
                # for i in range(1, 6):
                #     ind = (risk_acc[:, 2] < i) 
                #     risk_acc_cluster.append([np.sum(ind), math.sqrt(np.mean(risk_acc[ind, 3]) / 25) * 0.3048])
                # risk_acc_cluster.append(
                #     [np.sum(risk_acc[:, 2] > 5), math.sqrt(np.mean(risk_acc[risk_acc[:, 2] > 5, 3]) / 25) * 0.3048])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    args_dict = parser.parse_args()
    print(args_dict.dataset)

    writer = SummaryWriter('vals')
    names = ['12']
    evaluate = Evaluate()
    for epoch in names:
        evaluate.main(name=epoch, val=False, writer=False,dataset=args_dict.dataset)
