from torch.utils.data import DataLoader
import loader_risky_time as lo
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import os
from evaluate import Evaluate
from config import *
from torch.utils.tensorboard import SummaryWriter
import random
from torch.utils.data import Subset
from risk_decoder_autobot_v1_time import Risk_Decoder_auto_time, Criterion
from gnn_model_risk_large import *
from FoePredict import FoePredict_risky_time
import matplotlib.pyplot as plt
import matplotlib.cm as cm




def NLLLoss_score(pred_scores, pred, target, mask, maneuver, epoch, log_std_range=(-1.609, 5.0), rho_limit=0.5):
    batch_size = pred_scores.shape[0]
    # pred1 = pred[np.arange(batch_size),list(map(lambda x: 2*(x - 1), maneuver))]
    # pred2 = pred[np.arange(batch_size),list(map(lambda x: 2*x-1, maneuver))]
    # pred_new = t.stack((pred1, pred2), dim=1)
    distance = (pred[:, :, :, 0:2] - target[:, None, :, :]).norm(dim=-1)
    distance = (distance * mask[:, None, :]).sum(dim=-1)
    nearest_mode_idxs = distance.argmin(dim=-1)
    # preds_mode_idxs = pred_scores.argmax(dim=-1)
    nearest_mode_bs_idxs = t.arange(batch_size).type_as(nearest_mode_idxs)  # (batch_size, 2)
    nearest_trajs = pred[nearest_mode_bs_idxs, nearest_mode_idxs]

    # nearest_mode_idxs_fixed = [t + 2*(v-1) for t, v in zip(nearest_mode_idxs, maneuver)]
    # nearest_mode_idxs_fixed = t.stack(nearest_mode_idxs_fixed)

    nll_loss_sum = 0
    for k in range(pred_scores.shape[1]):
        traj_k = pred[:,k,:,:]
        mu_x, mu_y = traj_k[:, :, 0], traj_k[:, :, 1]
        sigma_x, sigma_y = traj_k[:, :, 2], traj_k[:, :, 3]
        rho = traj_k[:, :, 4]
        y_x, y_y = target[:, :, 0], target[:, :, 1]

        dx = y_x - mu_x
        dy = y_y - mu_y
        log_std1 = t.clip(sigma_x, min=log_std_range[0], max=log_std_range[1])
        log_std2 = t.clip(sigma_y, min=log_std_range[0], max=log_std_range[1])
        std1 = t.exp(log_std1)  # (0.2m to 150m)
        std2 = t.exp(log_std2)  # (0.2m to 150m)
        rho = t.clip(rho, min=-rho_limit, max=rho_limit)

        reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * t.log(1 - rho ** 2)  # (batch_size, num_timestamps)
        # prob_log = t.log(pred_scores[:,k]+1e-100)
        reg_gmm_exp = (0.5 * 1 / (1 - rho ** 2)) * (
                    (dx ** 2) / (std1 ** 2) + (dy ** 2) / (std2 ** 2) - 2 * rho * dx * dy / (
                        std1 * std2))#-prob_log 
        if epoch < 10:
            nll_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * mask).sum(dim=-1) * (pred_scores[:,k]+0.01)
        else:
            nll_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * mask).sum(dim=-1) * (pred_scores[:,k])
        nll_loss_sum += t.sum(nll_loss) / t.sum(mask[:,0]) 
    loss_cls = F.cross_entropy(input=pred_scores, target=nearest_mode_idxs, reduction='none')

    return nll_loss_sum, t.sum(loss_cls) / t.sum(mask[:,0]), nearest_trajs, nearest_mode_idxs



def MSELoss2(g_out, fut, mask, v_fut):
    # g_out = g_out.permute(1, 0, 2)
    muX = g_out[:, :, 0]
    muY = g_out[:, :, 1]
    x = fut[:, :, 0]
    y = fut[:, :, 1]
    vx = g_out[:, :, 5]
    vy = g_out[:, :, 6]
    vx_fut = v_fut[:, :, 0]
    vy_fut = v_fut[:, :, 1]
    out = t.pow(vx - vx_fut, 2) + t.pow(vy - vy_fut, 2)+t.pow(x - muX, 2) + t.pow(y - muY, 2)
    out = out * mask
    lossVal = t.sum(out) / t.sum(mask)
    return t.sum(out), t.sum(mask), lossVal





def MSELoss_goal(foe_pred, foe_gt, adj):
    out_pos = t.pow(foe_pred- foe_gt, 2)
    # out_ego = t.pow(foe_pred[:,0,:] - ego_future, 2)
    return t.sum(out_pos)  / t.sum(adj) #+t.sum(out_ego)

def MSELoss_goal_simple(foe_pred, foe_gt, adj):
    out_pos = t.pow(foe_pred- foe_gt, 2)
    return t.sum(out_pos) / t.sum(adj)


def CELoss(pred, target):
    value = t.log(t.sum(pred * target, dim=-1))
    return -t.sum(value) / value.shape[0]


def main():
    args['train_flag'] = True
    evaluate = Evaluate()
    gdEncoder_all = GDEncoder_all(args)
    generator = Risk_Decoder_auto_time(args)
    foe_predict = FoePredict_risky_time(args)
    l_path = args['path']
    # generator.load_state_dict(t.load(l_path + '/epoch4_g.tar'))  # , map_location='cuda:1'
    # foe_predict.load_state_dict(t.load(l_path + '/epoch4_f.tar'))
    # gdEncoder_all.load_state_dict(t.load(l_path + '/epoch4_gda.tar'))
    gdEncoder_all = gdEncoder_all.to(device)
    generator = generator.to(device)
    foe_predict = foe_predict.to(device)
    generator.train()
    gdEncoder_all.train()
    foe_predict.train()

    t1 = lo.RiskyDataset('shrp2/TrainSetRisky_all1.mat')

    seed = 23
    g = t.Generator()
    g.manual_seed(seed)
    trainDataloader = DataLoader(t1, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_worker'],
                                     collate_fn=t1.collate_fn, generator=g)

    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_foe = optim.Adam(foe_predict.parameters(), lr=learning_rate)
    optimizer_gd_all = optim.Adam(gdEncoder_all.parameters(), lr=learning_rate)

    scheduler_g = ExponentialLR(optimizer_g, gamma=0.6)
    scheduler_foe = ExponentialLR(optimizer_foe, gamma=0.6)
    scheduler_gd_all = ExponentialLR(optimizer_gd_all, gamma=0.6)
    criterion_loss = Criterion(args)

    
    for epoch in range(args['epoch']):
        print("epoch:", epoch + 1, 'lr', optimizer_g.param_groups[0]['lr'])
        
        loss_gi1 = 0
        loss_acc = 0
        loss_count = 0
        loss_goal1 = 0
        loss_goal_simple1 = 0
        loss_risk = 0

    
        for idx, data in enumerate(tqdm(trainDataloader)):
            hist, fut, mask,  grid_feature, adj, dsId, vehId, frame_Id, ego_current, v_fut, foe_future, foe_future_truth, maneuver = data
            #grid_feature: x,y,vx,vy,psi,dimx,dimy,ax,ay,distance_to_egox,distance_to_egoy,s_risk,o_risk
            # lane_change_ind = [index for index, value in enumerate(maneuver) if value != 1]
            # manueuver_list = manueuver_list+[(dsId[i], vehId[i][0], frame_Id[i], maneuver[i]) for i in lane_change_ind]
            grid_feature = grid_feature.to(device)
            # hist = hist.to(device)
            mask = mask.to(device)
            fut = fut.to(device)
            adj = adj.to(device)
            foe_future = foe_future.to(device)
            v_fut = v_fut.to(device)
            # fut_vel = fut[:, :, 2:4]
            # fut = fut[:, :, 0:2]

        
            values_all = gdEncoder_all(grid_feature, adj)
            # values = gdEncoder(grid_feature, adj, risk)
            foe_goal = foe_predict(grid_feature[:, :, :, :], adj[:, :], values_all[:,:,:,:]) ###grid_feature[:, 1:, :, 0:4], adj[:, 1:], values_all[:,1:,:,:]
            loss_goal = MSELoss_goal(foe_goal, foe_future[:, :, 1:, 0:4], adj[:, :]) ###adj[:, 1:]
            loss_goal_simple = MSELoss_goal_simple(foe_future[:, :,1:, 4:], foe_future[:, :, 1:, 0:4], adj[:, :])
            foe_destination = foe_goal.detach() 


            foe_destination = t.concat((foe_future[:, :, 0:1, 0:4], foe_destination), dim=2)
            g_out = generator(values_all[:,0,:,:], ego_current, foe_destination[:, :, :, :], grid_feature) ### values_all[:,0,:,:]
            # risk_loss = MSELoss_risk(risks_pred, risks_true)


            tmp = np.asarray(maneuver)
            maneuver = tmp[:,0].astype(int)
            loss_nll, loss_cls, predicted_trajectory, nearest_idx = NLLLoss_score(g_out['predicted_probability'], g_out['predicted_trajectory'], fut, mask, maneuver,epoch)
            ego_pos = fut[:,-1,t.arange(1, -1, -1)].unsqueeze(1)
            ego_vel =  v_fut[:,-1,t.arange(1, -1, -1)].unsqueeze(1)

  
            acc_sum, count_sum, loss_g1 = MSELoss2(predicted_trajectory, fut, mask, v_fut)
          
            loss_g = loss_nll + (loss_cls-1.2)*100 + loss_g1 + loss_goal/2 #+loss_stamp*100
            # risk_scale = (grid_feature[:, 0, 0, 11].cpu().sum().item()+grid_feature[:, 0, 0, 12].cpu().sum().item())/(len(grid_feature))
            # loss_g = loss_g*max(math.exp(risk_scale-6),1) #ngsim 6 highd 6cd
            # risk_scale = (grid_feature[:, 0, 0, 11].cpu().sum().item()+grid_feature[:, 0, 0, 12].cpu().sum().item())/(len(grid_feature))
            # loss_g = loss_g*max(math.exp(risk_scale-6.5),1)


            optimizer_g.zero_grad()
            optimizer_foe.zero_grad()
            optimizer_gd_all.zero_grad()

            loss_g.backward()
            if t.isnan(loss_g).any():
                print('loss_g is nan')
                continue

            a = t.nn.utils.clip_grad_norm_(generator.parameters(), 10)
            a = t.nn.utils.clip_grad_norm_(foe_predict.parameters(), 10)
            a = t.nn.utils.clip_grad_norm_(gdEncoder_all.parameters(), 10)

            optimizer_g.step()
            optimizer_foe.step()
            optimizer_gd_all.step()

            loss_gi1 += loss_g1.item()
            loss_goal1 += loss_goal.item()
            loss_goal_simple1 += loss_goal_simple.item()
   

            loss_acc += acc_sum.item()
            loss_count += count_sum.item()

        print('mse:', loss_gi1 / 5000)
        print('goal mse:', loss_goal1 / 5000)
        print('goal simple mse:', loss_goal_simple1 / 5000)
        print('rmse:', np.sqrt(loss_acc/loss_count)* 0.3048)

        # key_i=88
        # for idx, traj in enumerate(g_out['predicted_trajectory'][key_i]):
        #     color = cm.coolwarm(g_out['predicted_probability'][key_i][idx].cpu().detach().numpy())
        #     plt.plot(traj[:, 1].cpu().detach().numpy(), traj[:, 0].cpu().detach().numpy(), ':', color=color, linewidth=2)
        # plt.plot(fut[key_i,:, 1].cpu().detach().numpy(), fut[key_i,:, 0].cpu().detach().numpy(), '-', color='black', linewidth=2)
        # plt.savefig('trajectory_'+str(key_i)+'.png')
        # print('risk loss:', loss_risk / 5000)

        # writer.add_scalar('mse', loss_gi1 / 5000, epoch)


        save_model(name=str(epoch + 1), generator=generator, foe_predict=foe_predict, gdEncoder_all=gdEncoder_all)
        evaluate.main(name=str(epoch + 1), val=True, writer = False, dataset='shrp2')
        scheduler_g.step()
        scheduler_foe.step()
        scheduler_gd_all.step()


def save_model(name, generator, foe_predict, gdEncoder_all):
    l_path = args['path']
    if not os.path.exists(l_path):
        os.makedirs(l_path)
    t.save(generator.state_dict(), l_path + '/epoch' + name + '_g.tar')
    t.save(foe_predict.state_dict(), l_path + '/epoch' + name + '_f.tar')
    t.save(gdEncoder_all.state_dict(), l_path + '/epoch' + name + '_gda.tar')



if __name__ == '__main__':
    main()
