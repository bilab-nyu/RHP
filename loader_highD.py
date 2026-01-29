from __future__ import print_function, division
from torch.utils.data import Dataset
import scipy.io as scp
import numpy as np
import torch
import h5py
from config import args
import time
from scipy.spatial.distance import pdist, squareform
import time



class HighdDataset(Dataset):

    def __init__(self, track_file, traj_file, type, t_h=75, t_f=125, d_s=5, enc_size=64):
        
        # T2 = scp.loadmat(mat_file+'track_HighD_2.mat')['tracks']
        # self.T = np.concatenate((T1, T2), axis=0)
        if type == 'train':
            self.T  = scp.loadmat(track_file)['tracks']
            self.D  = scp.loadmat(traj_file)['trajTr']
        elif type == 'val':
            self.D = scp.loadmat(traj_file+'ValSetHighD.mat')['traj']
            T1  = scp.loadmat(track_file+'track_HighD_1.mat')['tracks']
            T2 = scp.loadmat(track_file+'track_HighD_2.mat')['tracks']
            self.T = np.concatenate((T1, T2), axis=0)
        elif type == 'test':
            self.D = scp.loadmat(traj_file+'TestSetHighD.mat')['traj']
            T1  = scp.loadmat(track_file+'track_HighD_1.mat')['tracks']
            T2 = scp.loadmat(track_file+'track_HighD_2.mat')['tracks']
            self.T = np.concatenate((T1, T2), axis=0)
        self.type = type
        self.t_h = t_h  #
        self.t_f = t_f  #
        self.d_s = d_s  # skip
        self.enc_size = enc_size  # size of encoder LSTM
        self.alltime = 0
        self.count = 0
        self.max_node = 30 + 1
        self.grid_feature_size = 13
        self.hist_len = self.t_h // self.d_s + 1
        self.fut_len = self.t_f // self.d_s
        self.endpoint_time = args['endpoint_time']




    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):##### add position encoding
        # start_time = time.time()   
        dsId = self.D[idx, 0]  # dataset id
        if self.type == 'train':
            dsId = (dsId - 1) % 15 + 1
        dsId = dsId.astype(int)
        vehId = self.D[idx, 1].astype(int)  # agent id
        t = self.D[idx, 2]  # frame
        grid = self.D[idx, 12:]

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist, gridinfo, refPos_hist, lane = self.getHistory(vehId, t, vehId, dsId)
        fut, refPos_fut, ego_fut, lane_fut = self.getFuture(vehId, t, dsId)
        if hist[-1, 1]-hist[0, 1] > 0:
            direction = 1
        else:
            direction = 0
        
        # if len(np.unique(lane)) > 1:
        #     print('lane change')
        adj = np.zeros([self.max_node])
        adj[0] = 1
        maneuver = 1
        # lon_enc = np.zeros([args['lon_length']])
        # lon_enc[int(self.D[idx, 10] - 1)] = 1
        # lat_enc = np.zeros([args['lat_length']])
        # lat_enc[int(self.D[idx, 9] - 1)] = 1

        grid_features = np.zeros((self.max_node, self.hist_len, self.grid_feature_size))
        foe_future = np.zeros((self.max_node, self.endpoint_time + 1, 8))

        
        
        if hist.size == 0 or fut.size == 0:
            mask = np.empty([0])
            return hist, fut, mask, grid_features, adj, dsId, [vehId], t, np.empty(
                [0, 4]),ego_fut, foe_future, [], [maneuver,np.inf,0,0]
        # elif direction == 1:
        #     return np.zeros_like(hist), np.zeros_like(fut), np.zeros_like(fut[:, 0]), grid_features, adj, dsId, [vehId], t, np.array(
        #         [0, 0, 1]),np.zeros_like(ego_fut), foe_future, [], [maneuver,np.inf,0,0]
        else:
            mask = np.ones_like(fut[:, 0])
        # gridinfo = self.getinfo(vehId, t, vehId, dsId)
        grid_features[0,:,0:9] = np.concatenate((hist, gridinfo), axis=1)

        if lane[0]>lane[-1] or lane[-1]>lane_fut:
            maneuver = 3
        elif lane[0]<lane[-1] or lane[-1]<lane_fut:
            maneuver = 2
        else:
            maneuver = 1

        histx = self.D[idx, 4] * 0.3048
        histy = self.D[idx, 3] * 0.3048
        vx = self.D[idx, 6] * 0.3048
        vy = self.D[idx, 5] * 0.3048
        dimx = self.D[idx, 8] * 0.3048
        dimy = self.D[idx, 9] * 0.3048
        ego_pos = [[histx, histy]]
        ego_vel = [[vx, vy]]
        ego_dim = [[dimx, dimy]]

        foe_pos = []
        foe_vel = []
        foe_dim = []
        vehIDs = [vehId]

        cnt=0
        foe_future_truth = []

        time_period = np.arange(self.t_f/self.endpoint_time,self.t_f+1,self.t_f/self.endpoint_time)/25
        fut_simplex = vx * time_period
        fut_simpley = vy * time_period
        fut_simplevx = np.ones(self.endpoint_time)*vx
        fut_simplevy = np.ones(self.endpoint_time)*vy
        foe_future[0,1:,4:]=np.stack((fut_simplex, fut_simpley, fut_simplevx, fut_simplevy),axis=1)
        fut_predx = fut[(time_period*5-1).astype(int),1]* 0.3048
        fut_predy = fut[(time_period*5-1).astype(int),0]* 0.3048
        vx_pred = ego_fut[(time_period*5-1).astype(int),1] * 0.3048
        vy_pred = ego_fut[(time_period*5-1).astype(int),0] * 0.3048
        foe_future[0,1:,0:4]=np.stack((fut_predx, fut_predy, vx_pred, vy_pred),axis=1)
        foe_future[0,0,2:4]=np.array([vx, vy])
        foe_future[0,0,6:8]=np.array([vx, vy])

        for i in grid:
            cnt+=1
            if i == 0:
                continue

            nbrsdis, gridinfo_nbrs, refPos, _ = self.getHistory(i.astype(int), t, vehId, dsId)
            if nbrsdis.size == 0:
                continue
            nbrsdis_relative = np.repeat(nbrsdis[-1:, :], self.hist_len, axis=0)
            nbrsdis = nbrsdis - nbrsdis_relative

            # gridinfo_nbrs = self.getinfo(i.astype(int), t, vehId, dsId)
            grid_features_nbrs = np.concatenate((nbrsdis, gridinfo_nbrs, nbrsdis_relative), axis=1)
            grid_features[cnt, :, 0:11] = grid_features_nbrs

            veh_traj = self.T[dsId - 1][int(i - 1)].transpose()
            traj_loc = (t-veh_traj[0, 0]).astype(int)#np.where(veh_traj[:, 0] == t)[0][0]

            histx = veh_traj[traj_loc, 2] * 0.3048
            histy = veh_traj[traj_loc, 1] * 0.3048
            vx = veh_traj[traj_loc, 4] * 0.3048
            vy = veh_traj[traj_loc, 3] * 0.3048
            dimx = veh_traj[traj_loc, 6] * 0.3048
            dimy = veh_traj[traj_loc, 7] * 0.3048

            foe_pos.append([histx, histy])
            foe_vel.append([vx, vy])
            foe_dim.append([dimx, dimy])

            vehIDs.append(i)

            traj_loc_fut = (t + self.t_f - veh_traj[0, 0]).astype(int)#np.where(veh_traj[:, 0] == t + self.t_f)[0]
            stpt = np.argwhere(veh_traj[:, 0] == t).item() + self.d_s
            fut_ind = np.argwhere(veh_traj[:, 0] == t).item() + self.t_f + 1
            enpt = np.minimum(len(veh_traj), fut_ind)

            if traj_loc_fut>len(veh_traj)-1:
                fut_predx_truth = histx + vx * np.arange(self.d_s,self.t_f+1,self.d_s)/25  # + 0.5 * foe_acc[i] * time_period ** 2
                fut_predy_truth = histy + vy * np.arange(self.d_s,self.t_f+1,self.d_s)/25  # + foe_acc[i] * time_period
                vx_truth = np.array([vx] * int(self.t_f/self.d_s))
                vy_truth = np.array([vy] * int(self.t_f/self.d_s))
                # psi_truth = np.array([psi]*int(self.t_f/self.d_s))
            else:
                # psi = veh_traj[traj_loc_fut[0], 5]
                fut_predx_truth = veh_traj[stpt:enpt:self.d_s, 2]* 0.3048
                fut_predy_truth = veh_traj[stpt:enpt:self.d_s, 1]* 0.3048
                vx_truth = veh_traj[stpt:enpt:self.d_s, 4] * 0.3048
                vy_truth = veh_traj[stpt:enpt:self.d_s, 3] * 0.3048
                # psi_truth = veh_traj[stpt:enpt:self.d_s, 5]
            
            fut_predx_truth = fut_predx_truth - refPos[1] * 0.3048
            fut_predy_truth = fut_predy_truth - refPos[0] * 0.3048

            # foe_future[cnt-1,4]=psi
            foe_future_truth.append(np.stack((fut_predx_truth, fut_predy_truth, vx_truth, vy_truth),axis=1))
            
            foe_future[cnt,0,0:4]=np.array([histx-refPos[1] * 0.3048, histy-refPos[0]*0.3048, vx, vy])
            foe_future[cnt,0,4:]=np.array([histx-refPos[1] * 0.3048, histy-refPos[0]*0.3048, vx, vy])
            for tt in range(self.endpoint_time):
                time_period = self.t_f / 25/self.endpoint_time * (tt + 1)
                fut_simplex = histx + vx * time_period - refPos[1] * 0.3048
                fut_simpley = histy + vy * time_period - refPos[0] * 0.3048
                fut_simplevx = vx
                fut_simplevy = vy
                foe_future[cnt,tt+1,4:]=np.array([fut_simplex, fut_simpley, fut_simplevx, fut_simplevy])
                traj_loc_fut1 = (t + self.t_f/self.endpoint_time * (tt + 1) - veh_traj[0, 0]).astype(int)
                # traj_loc_fut1 = np.where(veh_traj[:, 0] == t + self.t_f/self.endpoint_time * (tt + 1))[0]
                if traj_loc_fut1 > len(veh_traj)-1:
                    fut_predx = histx + vx * time_period  # + 0.5 * foe_acc[i] * time_period ** 2
                    fut_predy = histy + vy * time_period  # + foe_acc[i] * time_period
                    vx_pred = vx
                    vy_pred = vy
                else:
                    fut_predx = veh_traj[traj_loc_fut1, 2] * 0.3048
                    fut_predy = veh_traj[traj_loc_fut1, 1] * 0.3048
                    vx_pred = veh_traj[traj_loc_fut1, 4] * 0.3048
                    vy_pred = veh_traj[traj_loc_fut1, 3] * 0.3048
                fut_predx = fut_predx - refPos[1] * 0.3048
                fut_predy = fut_predy - refPos[0] * 0.3048
                foe_future[cnt,tt+1,0:4]=np.array([fut_predx, fut_predy,vx_pred, vy_pred])
            adj[cnt] = 1

        ego_current = np.concatenate((ego_pos[0], np.array([direction])))
        if len(foe_pos) == 0:
            return hist, fut, mask, np.asarray(grid_features), adj, dsId, [
                vehId], t, ego_current, ego_fut, foe_future, [], [maneuver,np.inf,0,0]

        ego_pos = np.repeat(ego_pos, len(foe_pos), axis=0)
        ego_vel = np.repeat(ego_vel, len(foe_vel), axis=0)
        ego_dim = np.repeat(ego_dim, len(foe_dim), axis=0)
        foe_dim = np.asarray(foe_dim)
        foe_pos = np.asarray(foe_pos)
        foe_vel = np.asarray(foe_vel)

        s_risk, o_risk = self.cal_risk(ego_pos, ego_vel, ego_dim, foe_pos, foe_vel, foe_dim)
        foe_risk = [[1, 1]]
        foe_risk.extend(np.stack((s_risk, o_risk), axis=1))
        foe_risks = np.asarray(foe_risk)

        ttc_min = 1.0 * np.inf
        risk_all = np.zeros((self.max_node, 2))
        if len(foe_pos) > 0:
            ttc = self.calculate_ttc(ego_pos, ego_vel, foe_pos, foe_vel, (ego_dim[:, 1]) / 2, (foe_dim[:, 1]) / 2)
            # ttc_reverse = 10/(ttc+1e-6)
            # foe_risk.extend(np.stack((ttc_reverse, ttc_reverse), axis=1))
            # foe_risks = np.asarray(foe_risk)
            ttc_min = np.array(np.min(ttc))
            risk_all[adj == 1, 0] = foe_risks[:, 0]
            risk_all[adj == 1, 1] = foe_risks[:, 1]
            risk_all[0, :] = np.sum(foe_risks[1:, :], axis=0)
            # risk_all1 = np.repeat(risk_all[:,1:2], repeats=2, axis=1)
            grid_features[:, :, 11:] = np.tile(risk_all, (1, self.hist_len)).reshape(self.max_node, self.hist_len, 2)
            

       
         
        # end_time = time.time()  
        # elapsed_time = end_time - start_time
        # print('time 2:', elapsed_time)



        return hist, fut, mask, grid_features, adj, dsId, vehIDs, t, ego_current, ego_fut, foe_future, foe_future_truth, [maneuver,ttc_min, risk_all[0, 0], risk_all[0, 1]]
    

    def get_neighbor_fut(self, vehId, t, dsId):
        foe_future = []
        for ds in range(len(dsId)):
            vehId_ego = vehId[ds][0]
            vehId_i = vehId[ds][1:]
            foe_future_i = np.zeros((len(vehId_i),self.fut_len, 4))
            ds_i = dsId[ds]
            t_i = t[ds]
            for idx, i in enumerate(vehId_i):
                veh_traj = self.T[ds_i - 1][int(i - 1)].transpose()
                traj_loc = np.where(veh_traj[:, 0] == t_i)[0][0]
                histx = veh_traj[traj_loc, 2] * 0.3048
                histy = veh_traj[traj_loc, 1] * 0.3048
                vx = veh_traj[traj_loc, 4] * 0.3048
                vy = veh_traj[traj_loc, 3] * 0.3048
                stpt = np.argwhere(veh_traj[:, 0] == t_i).item() + self.d_s
                fut_ind = np.argwhere(veh_traj[:, 0] == t_i).item() + self.t_f + 1
                enpt = np.minimum(len(veh_traj), fut_ind)

                if fut_ind > len(veh_traj):
                    fut_predx = histx + vx * np.arange(self.d_s,self.t_f+1,self.d_s)/25  # + 0.5 * foe_acc[i] * time_period ** 2
                    fut_predy = histy + vy * np.arange(self.d_s,self.t_f+1,self.d_s)/25  # + foe_acc[i] * time_period
                    vx = np.array([vx] * int(self.t_f/self.d_s))
                    vy = np.array([vy] * int(self.t_f/self.d_s))
                else:
                    fut_predx = veh_traj[stpt:enpt:self.d_s, 2]* 0.3048
                    fut_predy = veh_traj[stpt:enpt:self.d_s, 1]* 0.3048
                    vx = veh_traj[stpt:enpt:self.d_s, 4] * 0.3048
                    vy = veh_traj[stpt:enpt:self.d_s, 3] * 0.3048
                nbrsdis, refPos, _ = self.getHistory(i.astype(int), t_i, vehId_ego, ds_i)
                fut_predx = fut_predx - refPos[1] * 0.3048
                fut_predy = fut_predy - refPos[0] * 0.3048

                foe_future_i[idx, :,0:2] = np.stack((fut_predx, fut_predy),axis=1)
                foe_future_i[idx, :,2:4] = np.stack((vx, vy),axis=1)
            foe_future.append(foe_future_i)
        return foe_future

 
    def cal_risk(self, ego_pos, ego_vel, ego_dim, foe_pos, foe_vel, foe_dim):
        dx = ego_pos[:, 0] - foe_pos[:, 0]
        dy = ego_pos[:, 1] - foe_pos[:, 1]
        epsilon = 1e-2
        # get dx and dy WITH vehicle shapes
        dx = np.clip(np.abs(dx) - 0.5 * (ego_dim[:, 0] + foe_dim[:, 0]), 0, None) * dx / (abs(dx) + epsilon)
        dy = np.clip(np.abs(dy) - 0.5 * (ego_dim[:, 1] + foe_dim[:, 1]), 0, None) * dy / (abs(dy) + epsilon)
        # get relative speeds (+ if distance is decreasing, else -)
        dVx = -((ego_vel[:, 0]) - (foe_vel[:, 0])) * dx / (abs(dx) + epsilon)
        dVy = -(ego_vel[:, 1] - foe_vel[:, 1]) * dy / (abs(dy) + epsilon)

        gx = np.abs(ego_vel[:, 0]) * 1.5000 + 2
        gy = np.abs(ego_vel[:, 0]) * 0.03 + 1
        bx = np.abs(ego_vel[:, 0]) * 0.04 + 2
        by = 5.5
        # s_risk = np.exp(-np.abs(dx / gx) ** bx - np.abs(dy / gy) ** by+2)
        s_risk = np.exp(-2*(np.abs(dx / gx) ** bx + np.abs(dy / gy) ** by - 1)) 

        # assume we have these parameters for 'objective field'
        _d = 3
        _t = 5
        _b1 = 5
        _b2 = 3
        alpha = 2
        beta = 2
        # define an epsilon to prevent 0 distance
        epsilon_v = 1e-4
        # define an epsilon to set negative times to a large number
        epsilon_t = 1e-4
        # get o_risks
        v = np.sqrt(dVx ** 2 + dVy ** 2)
        # add a mask to prevent risk=1 when v=0 where x!=0
        v_mask = (v != 0) & (dx ** 2 + dy ** 2 > 0)
        v = np.clip(v, epsilon_v, None)
        dm2 = (dVx * dy - dVy * dx) ** 2 / (v ** 2)
        td1 = -(dVx * dx + dVy * dy) / (v ** 2)
        td = 1 / np.clip(1 / (td1+1e-6), epsilon_t, None)
        o_risk = np.exp(-alpha * (dm2 / _d) ** _b1 + 1) * np.exp(-beta * (td / _t) ** _b2 + 1) * v_mask
        return s_risk, o_risk


    def calculate_ttc(self, pos1, vel1, pos2, vel2, radius1, radius2):
        """
        Calculate Time to Collision (TTC) for two objects in 2D space.

        Parameters:
        pos1: numpy array, position of object 1 [x, y]
        vel1: numpy array, velocity of object 1 [vx, vy]
        pos2: numpy array, position of object 2 [x, y]
        vel2: numpy array, velocity of object 2 [vx, vy]
        radius1: float, radius of object 1
        radius2: float, radius of object 2

        Returns:
        float: Time to collision (TTC) in seconds, or float('inf') if no collision
        """
        # Relative position and velocity
        rel_pos = pos2 - pos1
        rel_vel = vel2 - vel1

        # Combined radius
        r = radius1 + radius2

        # Coefficients for quadratic equation: a*t^2 + b*t + c = 0
        a = np.sum(rel_vel * rel_vel, axis=1)  # dot product of relative velocity
        b = 2 * np.sum(rel_pos * rel_vel, axis=1)  # 2 * dot product of relative position and velocity
        c = np.sum(rel_pos * rel_pos, axis=1) - r ** 2  # adjusted for radii

        # Initialize TTC array
        ttc = np.full(a.shape, np.inf)

        # Avoid division by zero where relative velocity is zero
        non_zero_a = np.abs(a) > 1e-10

        # Calculate discriminant
        discriminant = b[non_zero_a] ** 2 - 4 * a[non_zero_a] * c[non_zero_a]

        # Only consider cases with real roots (discriminant >= 0)
        valid = discriminant >= 0
        valid_indices = np.where(non_zero_a)[0][valid]

        # Solve for positive roots
        sqrt_disc = np.sqrt(discriminant[valid])
        a_valid = a[valid_indices]
        b_valid = b[valid_indices]

        # Calculate possible collision times
        t1 = (-b_valid + sqrt_disc) / (2 * a_valid)
        t2 = (-b_valid - sqrt_disc) / (2 * a_valid)

        # Select the smallest positive time
        for i, idx in enumerate(valid_indices):
            times = [t for t in [t1[i], t2[i]] if t > 0]
            if times:
                ttc[idx] = min(times)

        return ttc


    def getinfo(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 7])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 7])
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 7])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 3:10]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 7])
            return hist



    ## Helper function to get track history
    def getHistory(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2]), np.empty([0, 7]), np.zeros([2]), np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2]), np.empty([0, 7]), np.zeros([2]), np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            x = (t-refTrack[0, 0]).astype(int)#np.where(refTrack[:, 0] == t)
            refPos = refTrack[x, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2]), np.empty([0, 7]), np.empty([0, 2]), np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
                hist_info = vehTrack[stpt:enpt:self.d_s, 3:10]
                lane = vehTrack[stpt:enpt:self.d_s, 5]
            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2]), np.empty([0, 7]), np.zeros([2]), np.empty([0, 1])
            return hist, hist_info, refPos, lane



    ## Helper function to get track future
    def getFuture(self, vehId, t, dsId):
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos = vehTrack[(t-vehTrack[0, 0]).astype(int), 1:3]#np.where(vehTrack[:, 0] == t)
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        lane = vehTrack[enpt-self.d_s, 5]
        fut_endpoint = vehTrack[enpt - 1, 1:3] - refPos
        ego_fut = vehTrack[stpt:enpt:self.d_s, 3:5]


        return fut, refPos, ego_fut, lane

    ## Collate function for dataloader
    def collate_fn(self, samples):

        # Initialize social mask batch:
        frameId_batch = []
        dsId_batch = []
        vehId_batch = []
        grid_feature_batch = torch.zeros(len(samples), self.max_node, self.hist_len, self.grid_feature_size)##
        hist_traj_batch = torch.zeros(len(samples), self.hist_len, 2)
        fut_traj_batch = torch.zeros(len(samples), self.fut_len, 2)
        mask_fut_batch = torch.zeros(len(samples), self.fut_len).bool()
        # refPos_hist_batch = torch.zeros(len(samples), 2)
        foe_future_truth_batch = []
        foe_future_batch = torch.zeros(len(samples), self.max_node, self.endpoint_time+1, 8)
        ego_current_batch = torch.zeros(len(samples), 3)
        adj_batch = torch.zeros(len(samples), self.max_node)
        ego_future_batch = torch.zeros(len(samples), self.fut_len, 2)
        maneuver_batch = []


        for sampleId, (hist, fut, mask,grid_features, adj, dsId, vehIDs, t, ego_current,ego_fut, foe_future, foe_future_truth, maneuver) in enumerate(samples):
            dsId_batch.append(dsId)
            vehId_batch.append(vehIDs)
            frameId_batch.append(t)
            maneuver_batch.append(maneuver)
            # refPos_hist_batch[sampleId] = torch.from_numpy(refPos_hist)
            grid_feature_batch[sampleId] = torch.from_numpy(grid_features)
            hist_traj_batch[sampleId] = torch.from_numpy(hist)
            adj_batch[sampleId] = torch.from_numpy(adj)
            fut_traj_batch[sampleId, 0:len(fut), :] = torch.from_numpy(fut)
            mask_fut_batch[sampleId, 0:len(mask)] = torch.from_numpy(mask)
            foe_future_truth_batch.append(foe_future_truth)
            foe_future_batch[sampleId] = torch.from_numpy(foe_future)
            ego_current_batch[sampleId] = torch.from_numpy(ego_current)
            ego_future_batch[sampleId,0:len(ego_fut)] = torch.from_numpy(ego_fut)

        return (hist_traj_batch, fut_traj_batch, mask_fut_batch, grid_feature_batch,
                adj_batch, dsId_batch, vehId_batch, frameId_batch, ego_current_batch, ego_future_batch, foe_future_batch, foe_future_truth_batch, maneuver_batch)


