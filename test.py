import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import argparse
import random
# import itertools
import h5py
from script.nets import *
from script.utils import *
from script.models import *

def get_args(argv=None):
    parser = argparse.ArgumentParser(description = 'Put your hyperparameters')

    parser.add_argument('-dp', '--data_path', default='dt_0.01_fr_1.0', type=str, help='data path name')
    parser.add_argument('-lf', '--logs_fname', default='new_physics_8', type=str, help='logs file name')
    parser.add_argument('-dc', '--dict', default='nse', type=str, help='dict name')
    parser.add_argument('-dr', '--data_rate', default=1, type=float, help='logs file name')
    
    parser.add_argument('-L', '--L', default=4, type=int, help='the number of layers')
    parser.add_argument('-m', '--modes', default=12, type=int, help='the number of modes of Fourier layer')
    parser.add_argument('-w', '--width', default=32, type=int, help='the number of width of FNO layer')
    parser.add_argument('--drop', default=0.1, type=float, help='Dropout of unet')
    parser.add_argument('-ch', '--channel', default=32, type=int, help='channels of unet')
    
    parser.add_argument('--phys_gap', default=40, type=int, help = 'Number of gap of Phys')
    parser.add_argument('--phys_epochs', default=10, type=int, help = 'Number of Phys Epochs')
    parser.add_argument('--phys_steps', default=2, type=int, help = 'Number of Phys Steps')
    parser.add_argument('--phys_scale', default=0.1, type=float, help = 'Number of Phys Scale')
    parser.add_argument('--phys_random_select', default=False, type=bool, help = 'Whether random select')

    parser.add_argument('--batch_size', default=32, type=int, help = 'batch size')
    parser.add_argument('--epochs', default=100, type=int, help = 'Number of Epochs')
    parser.add_argument('--re_epochs', default=0, type=int, help = 'Number of Phys Retrain Epochs')
    parser.add_argument('--tune_epochs', default=0, type=int, help = 'Number of Data Retrain Epochs')
    parser.add_argument('--loop', default=2, type=int, help = 'Number of Total Loop')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--step_size', default=100, type=int, help='scheduler step size')
    parser.add_argument('--gamma', default=0.5, type=float, help='scheduler factor')
    parser.add_argument('--gpu', default=0, type=int, help='device number')

    parser.add_argument('-tg', '--tg', default=1, type=int, help = 'time gap')
    parser.add_argument('-Ng', '--Ng', default=1, type=int, help = 'N gap')
    parser.add_argument('-l1', '--lambda1', default=1, type=float, help='weight of data losses1')
    parser.add_argument('-l2', '--lambda2', default=0, type=float, help='weight of encode losses1')
    parser.add_argument('-l3', '--lambda3', default=0, type=float, help='weight of encode losses2')
    parser.add_argument('-l4', '--lambda4', default=0, type=float, help='weight of physics losses')
    parser.add_argument('-l5', '--lambda5', default=0, type=float, help='weight of physics losses of encoder')
    parser.add_argument('-l6', '--lambda6', default=0, type=float, help='weight of hidden loss')
    parser.add_argument('-fc', '--f_channels', default=0, type=int, help='channels of f encode')
    
    parser.add_argument('--re', default=0.1, type=float, help='re number')
    parser.add_argument('--lx', default=2.05, type=float, help='Lx')
    parser.add_argument('--ly', default=2.05, type=float, help='Ly')
    parser.add_argument('--en_dim', default=3, type=int, help='the input dimension of first layer of encoder')
    parser.add_argument('--gap_sample', default=5, type=int, help='sample gap from origin uvp in space')
    parser.add_argument('--obs_num', default=7, type=int, help='width and length of observation')
    parser.add_argument('--state_num', default=32, type=int, help='width and length of of state')
    parser.add_argument('--recall_size', default=4, type=int, help='recall history observation')

    parser.add_argument('--with_next', default=1, type=int, help='with ut+1 or not')
    parser.add_argument('--start_phys', default=500, type=int, help='with ut+1 or not')
    parser.add_argument('--seed', default=333, type=int, help='set random seed')
    parser.add_argument('--traj_str_tr', default=0, type=int, help='select the train set start')
    parser.add_argument('--traj_end_tr', default=400, type=int, help='select the train set end')
    parser.add_argument('--traj_str_ts', default=450, type=int, help='select the test set start')
    parser.add_argument('--traj_end_ts', default=500, type=int, help='select the test set end')

    parser.add_argument('-trsta', '--train_str', default=0, type=int, help='start position train')
    parser.add_argument('-trend', '--train_end', default=14, type=int, help='end position train')
    parser.add_argument('-tssta', '--test_str', default=0, type=int, help='start position test')
    parser.add_argument('-tsend', '--test_end', default=14, type=int, help='end position test')

    parser.add_argument('--unlabel_size', default=1/4, type=float, help='select the unlabel set scale')

    parser.add_argument('--ups', default=1, type=int, help='how to upsample in unet')
    parser.add_argument('--full', default=False, type=int, help='if the trans phys loss backward to encoder')

    return parser.parse_args(argv)

if __name__=='__main__':
    # args parser
    args = get_args()
    # args = torch.load('arg')
    print(args)
    # torch.save(args, 'arg')
    # assert 1 == 2
    # logs
    logs = dict()
    logs['args'] = args
    logs['mean'] = []
    logs['std'] = []
    logs['full'] = args.full

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    

    logs_fname = f'/users/swe_code/logs/model_{args.dict}/phase_{args.logs_fname}'

    data_path = '/users/swe_data/swe_part_data'
    full_data_path = '/users/swe_data/swe_data'
    
    data_run = torch.load(data_path, map_location=device)
    full_data = torch.load(full_data_path, map_location=device)  # uvp, traj, t, x, y
    
    num_obs = args.obs_num
    num_state = args.state_num
    gap = args.gap_sample
    trsta = args.train_str
    trend = args.train_end
    tssta = args.test_str
    tsend = args.test_end
    traj_str_tr = args.traj_str_tr
    traj_end_tr = args.traj_end_tr
    traj_str_ts = args.traj_str_ts
    traj_end_ts = args.traj_end_ts
    unlabel_size = args.unlabel_size

    data_train = data_run[:,traj_str_tr:traj_end_tr]  # uvp, traj, t, ox*oy
    full_train = full_data[:,traj_str_tr:traj_end_tr]  # uvp, traj, t, x, y
    data_test = data_run[:,traj_str_ts:traj_end_ts]
    full_test = full_data[:,traj_str_ts:traj_end_ts]

    data_train = data_train.permute(1,0,2,3)  # traj, uvp, t, ox*oy
    data_train = data_train[:,:,trsta:trend]
    data_test = data_test.permute(1,0,2,3)  # traj, uvp, t, ox*oy
    data_test = data_test[:,:,tssta:tsend]

    full_train = full_train.permute(1,0,2,3,4)  # traj, uvp, t, ox*oy
    full_train = full_train[:,:,trsta:trend]
    full_test = full_test.permute(1,0,2,3,4)  # traj, uvp, t, ox*oy
    full_test = full_test[:,:,tssta:tsend]

    data_run_train, data_run_test, full_train, full_test, logs = normal(data_train, data_test, full_train, full_test, logs)  # traj, uvp, t, ox*oy

    data_run_train = data_run_train.permute(1,0,2,3)  # uvp, traj, t, ox*oy
    data_run_test = data_run_test.permute(1,0,2,3)

    data_train_re = data_repeat(data_run_train, num_obs, num_state, gap)  # uvp, t, traj, x, y
    data_test_re = data_repeat(data_run_test, num_obs, num_state, gap)
    
    data_train_re = data_train_re.permute(2,0,1,3,4)  # traj, uvp, t, x, y
    data_test_re = data_test_re.permute(2,0,1,3,4)

    data_run = torch.cat((data_train_re, full_train.to(data_train_re.device)), axis=1)  # traj, uvp*2, t, x, y
    data_test = torch.cat((data_test_re, full_test.to(data_test_re.device)), axis=1)  # traj, uvp*2, t, x, y

    tg = args.tg
    Ng = args.Ng 
    batch_size=args.batch_size
    dt = tg*0.01
    
    ctr_boun_ini = torch.tensor(np.zeros((batch_size, data_run.shape[3], data_run.shape[4], 1)), dtype=torch.float64)  # (16,41,41,1)
    N0, nt, nx, ny = data_run.shape[0], data_run.shape[2], data_run.shape[3], data_run.shape[4]
    shape = [nx, ny]
    
    recall_size = args.recall_size
    input_data =data_run[:, :, :-1, :]  # traj, uvp*2, t, x, y
    output_data = data_run[:, :, 1:, :]
    
    input_data_test = data_test[:, :, :-1, :]
    output_data_test = data_test[:, :, 1:, :]

    data_in = input_data.permute(0,2,1,3,4)  # traj, t, uvp*2, x, y
    labels_out = output_data.permute(0,2,1,3,4)

    data_in_ensmble = np.empty((data_in.shape[0],data_in.shape[1]-recall_size+1,recall_size*6,data_in.shape[-2],data_in.shape[-1]))
    labels_out_ensmble = np.empty((labels_out.shape[0],labels_out.shape[1]-recall_size+1,recall_size*6,labels_out.shape[-2],labels_out.shape[-1]))
    
    data_in_test = input_data_test.permute(0,2,1,3,4)  # # traj, t, uvp*2, x, y
    labels_out_test = output_data_test.permute(0,2,1,3,4)
    data_in_test_ensmble = np.empty((data_in_test.shape[0],data_in_test.shape[1]-recall_size+1,recall_size*6,data_in_test.shape[-2],data_in_test.shape[-1]))
    labels_out_test_ensmble = np.empty((labels_out_test.shape[0],labels_out_test.shape[1]-recall_size+1,recall_size*6,labels_out_test.shape[-2],labels_out_test.shape[-1]))

    for i in range(data_in_ensmble.shape[1]):
        data_in_ensmble[:,i] = data_in[:,i:i+recall_size].reshape(-1,recall_size*6,data_in.shape[-2],data_in.shape[-1])
        labels_out_ensmble[:,i] = labels_out[:,i:i+recall_size].reshape(-1,recall_size*6,labels_out.shape[-2],labels_out.shape[-1])
        
    for i in range(data_in_test_ensmble.shape[1]):
        data_in_test_ensmble[:,i] = data_in_test[:,i:i+recall_size].reshape(-1,recall_size*6,data_in_test.shape[-2],data_in_test.shape[-1])
        labels_out_test_ensmble[:,i] = labels_out_test[:,i:i+recall_size].reshape(-1,recall_size*6,labels_out_test.shape[-2],labels_out_test.shape[-1])
    
    train_data = torch.tensor(data_in_ensmble, dtype=torch.float64)  # traj, t_en, uvp*2*n, x, y
    train_labels = torch.tensor(labels_out_ensmble, dtype=torch.float64)
       
    unlabel_data = train_data[-int(N0*unlabel_size):]
    unlabel_lab = train_labels[-int(N0*unlabel_size):]
    train_data = train_data[:-int(N0*unlabel_size)]
    train_labels = train_labels[:-int(N0*unlabel_size)]
    
    unlabel_data = unlabel_data.view(-1, unlabel_data.shape[2], unlabel_data.shape[3], unlabel_data.shape[4])
    unlabel_lab = unlabel_lab.view(-1, unlabel_lab.shape[2], unlabel_lab.shape[3], unlabel_lab.shape[4])
    train_data = train_data.view(-1, train_data.shape[2], train_data.shape[3], train_data.shape[4])
    train_labels = train_labels.view(-1, train_labels.shape[2], train_labels.shape[3], train_labels.shape[4])

    retrain_dataset = MyDataset(unlabel_data, unlabel_lab)
    train_dataset = MyDataset(train_data, train_labels)
    retrain_dataloader = DataLoader(retrain_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    test_data = torch.tensor(data_in_test_ensmble, dtype=torch.float64)
    test_labels = torch.tensor(labels_out_test_ensmble, dtype=torch.float64)
    test_data = test_data.view(-1, test_data.shape[2], test_data.shape[3], test_data.shape[4])
    test_labels = test_labels.view(-1, test_labels.shape[2], test_labels.shape[3], test_labels.shape[4])
    test_dataset = MyDataset(test_data, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


    logs['pred_model'] = []
    logs['phys_inform_model'] = []

    logs['test_loss_pre'] = []
    logs['test_loss_state'] = []
    logs['test_loss_ctr'] = []
    logs['test_loss_outen'] = []
    logs['test_loss_encode'] = []
    logs['test_loss_phys'] = []
    logs['train_loss_pre'] = []
    logs['train_loss_state'] = []
    logs['train_loss_ctr'] = []
    logs['train_loss_outen'] = []
    logs['train_loss_encode'] = []
    logs['train_loss_phys'] = []
    logs['train_loss_pde_phys'] = []

    logs['ori_data_loss'] = []
    logs['gen_ctr_data_mean'] = []
    logs['gen_ctr_data_std'] = []
    logs['gen_in_data_mean'] = []
    logs['gen_in_data_std'] = []

    logs['gen_data_loss'] = []
    logs['ori_ctr_data_mean'] = []
    logs['ori_ctr_data_std'] = []
    logs['ori_in_data_mean'] = []
    logs['ori_in_data_std'] = []

    logs['compare_error'] = []
    logs['baseline_error'] = []

    logs['mod'] = []
    logs['Cd_loss'] = []
    logs['Cl_loss'] = []
    logs['out_loss'] = []
    logs['abs_loss'] = []

    logs['test_compare_error'] = []

    logs['train_data'] = []
    logs['test_data'] = []
    
    logs['test_encoder_input'] = []
    logs['test_encoder_output'] = []
    
    logs['train_pred_loss'] = []
    logs['train_input_en_loss'] = []
    logs['train_output_en_loss'] = []
    logs['train_physics_loss'] = []
    logs['train_phys_encode_loss'] = []
    logs['train_hidden_loss'] = []
    logs['train_error_pred'] = []
    logs['train_error_input'] = []
    logs['train_error_output'] = []
    
    logs['test_pred_loss'] = []
    logs['test_input_en_loss'] = []
    logs['test_output_en_loss'] = []
    logs['test_physics_loss'] = []
    logs['test_phys_encode_loss'] = []
    logs['test_hidden_loss'] = []
    logs['test_error_pred'] = []
    logs['test_error_input'] = []
    logs['test_error_output'] = []

    logs['retrain_phys_loss'] = []
    logs['retrain_data_loss'] = []


    # model setting
    nse_model = NSEModel_FNO(shape, dt, args)
    params_num = nse_model.count_params()

    print('N0: {}, nt: {}, nx: {}, ny: {}, device: {}'.format(N0, nt, nx, ny, nse_model.device))
    print(f'param numbers of the model: {params_num}')

    # train process
    for loop in range(1,nse_model.params.loop+1):
        for epoch in range(1, nse_model.params.epochs+1):
            nse_model.phys_inform(epoch, train_dataloader, ctr_boun_ini, logs)
            if epoch % 1 == 0:
                nse_model.phys_test(test_dataloader, ctr_boun_ini, logs)
                
        for epoch in range(1, nse_model.params.re_epochs+1):
            nse_model.phys_retrain(epoch, retrain_dataloader, ctr_boun_ini, logs)
            if epoch % 1 == 0:
                nse_model.phys_test(test_dataloader, ctr_boun_ini, logs)

        for epoch in range(1, nse_model.params.tune_epochs+1):
            nse_model.data_retrain(epoch, train_dataloader, ctr_boun_ini, logs)
            if epoch % 1 == 0:
                nse_model.phys_test(test_dataloader, ctr_boun_ini, logs)

    torch.save([nse_model.phys_inform_model.state_dict(), logs], logs_fname)
