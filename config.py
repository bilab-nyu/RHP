args = {}
import random
import numpy as np
import torch as t
args['path'] = 'checkpoint/shrp2'
# -------------------------------------------------------------------------

seed = 72
random.seed(seed)
np.random.seed(seed)
t.manual_seed(seed)
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

learning_rate = 0.0005
args['dataset'] = "shrp2"  # highd shrp2
args['ego_goals_path'] = 'data/shrp2/' ###shrp2 highd
args['num_worker'] = 8
args['device'] = device
args['lstm_encoder_size'] = 64
args['n_head'] = 4
args['att_out'] = 48
args['in_length'] = 16
args['out_length'] = 25
args['f_length'] = 13 ##### ngsim 12 highd 13
args['traj_linear_hidden'] = 32
args['batch_size'] = 128
args['use_elu'] = True
args['dropout'] = 0
args['relu'] = 0.1
args['lat_length'] = 3
args['lon_length'] = 3  # 2
args['use_true_man'] = False
args['epoch'] = 20
args['use_spatial'] = False
args['grid_size'] = 39
args['goal_modes'] = 100
args['c'] = 6
args['tx_hidden_size'] = 384
args['endpoint_time'] = 5


args['use_maneuvers'] = False

args['cat_pred'] = True
args['use_mse'] = True
args['pre_epoch'] = 0
args['val_use_mse'] = True

args['entropy_weight'] = 40.0
args['kl_weight'] = 20.0
args['use_FDEADE_aux_loss'] = False


# -------------------------------------------------------------------------
