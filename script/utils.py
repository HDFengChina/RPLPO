from tkinter import N
import torch
import operator
import numpy as np
import os, sys
from functools import reduce
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import yaml
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def ensemble(input_data, output_data, recall_size):
    data_in = input_data.permute(1,0,2,3)
    labels_out = output_data.permute(1,0,2,3)
    data_in_ensmble = np.empty((data_in.shape[0]-recall_size,recall_size*6,data_in.shape[-2],data_in.shape[-1]))
    labels_out_ensmble = np.empty((labels_out.shape[0]-recall_size,recall_size*6,labels_out.shape[-2],labels_out.shape[-1]))
    for i in range(data_in_ensmble.shape[0]):
        data_in_ensmble[i] = data_in[i:i+recall_size].reshape(-1,data_in.shape[-2],data_in.shape[-1])  # (nt, 3*recall, 41, 41)
        labels_out_ensmble[i] = labels_out[i:i+recall_size].reshape(-1,labels_out.shape[-2],labels_out.shape[-1])  # (nt, 3*recall, 41, 41)
    return data_in_ensmble, labels_out_ensmble

def size(data_run):
    N0, nt, nx, ny = 1, data_run.shape[1], data_run.shape[2],data_run.shape[3]
    shape = [nx, ny]
    return shape, N0, nt, nx, ny

def data_divide(data_run):
    input_data =torch.tensor(data_run[:, :, :-1, :])
    output_data = torch.tensor(data_run[:, :, 1:, :])
    return input_data, output_data

def normal(data_run_train, data_run_test, full_run_train, full_run_test, logs):

    u_mean, u_std = data_run_train[:,0,:,:].mean(), data_run_train[:,0,:,:].std()
    v_mean, v_std = data_run_train[:,1,:,:].mean(), data_run_train[:,1,:,:].std()
    h_mean, h_std = data_run_train[:,2,:,:].mean(), data_run_train[:,2,:,:].std()
    logs['mean'].append(u_mean)
    logs['mean'].append(v_mean)
    logs['mean'].append(h_mean)
    logs['std'].append(u_std)
    logs['std'].append(v_std)
    logs['std'].append(h_std)

    data_run_train[:,0,:,:] = (data_run_train[:,0,:,:] - u_mean) / u_std
    data_run_train[:,1,:,:] = (data_run_train[:,1,:,:] - v_mean) / v_std
    data_run_train[:,2,:,:] = (data_run_train[:,2,:,:] - h_mean) / h_std
    full_run_train[:,0,:,:,:] = (full_run_train[:,0,:,:,:] - u_mean) / u_std
    full_run_train[:,1,:,:,:] = (full_run_train[:,1,:,:,:] - v_mean) / v_std
    full_run_train[:,2,:,:,:] = (full_run_train[:,2,:,:,:] - h_mean) / h_std

    data_run_test[:,0,:,:] = (data_run_test[:,0,:,:] - u_mean) / u_std
    data_run_test[:,1,:,:] = (data_run_test[:,1,:,:] - v_mean) / v_std
    data_run_test[:,2,:,:] = (data_run_test[:,2,:,:] - h_mean) / h_std
    full_run_test[:,0,:,:,:] = (full_run_test[:,0,:,:,:] - u_mean) / u_std
    full_run_test[:,1,:,:,:] = (full_run_test[:,1,:,:,:] - v_mean) / v_std
    full_run_test[:,2,:,:,:] = (full_run_test[:,2,:,:,:] - h_mean) / h_std


    return data_run_train, data_run_test, full_run_train, full_run_test, logs

def load_config(file):
    with open(file, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    return config

def rel_error(x, _x):
    """
    <ARGS>
    x : torch.Tensor shape of (B, *)
    _x : torch.Tensor shape of (B, *)
    <RETURN>
    out :torch.Tensor shape of (B), batchwise relative error between x and _x : (||x-_x||_2/||_x||_2)
    
    """
    if len(x.shape)==1:
        x = x.reshape(1, -1)
        _x = _x.reshape(1, -1)
    else:
        B = x.size(0)
        x, _x = x.reshape(B, -1), _x.reshape(B, -1)

    return torch.norm(x - _x, 2, dim=1)/torch.norm(_x, 2, dim=1)

def abs_error(x, _x):
    """
    <ARGS>
    x : torch.Tensor shape of (B, *)
    _x : torch.Tensor shape of (B, *)
    <RETURN>
    out :torch.Tensor shape of (B), batchwise relative error between x and _x : (||x-_x||_2/||_x||_2)
    
    """
    if len(x.shape)==1:
        x = x.reshape(1, -1)
        _x = _x.reshape(1, -1)
    else:
        B = x.size(0)
        x, _x = x.reshape(B, -1), _x.reshape(B, -1)
    
    return torch.norm(x - _x, 2, dim=1) 

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

def masker(sample_gap, data):
    out_data = data
    weight_loss = torch.tensor(np.zeros((out_data.shape[0], out_data.shape[1], out_data.shape[2])))
    for i in range(out_data.shape[1]):
        for z in range(out_data.shape[2]):
            if z%sample_gap == 0 and i%sample_gap == 0:
                weight_loss[:,i,z] = 0
            else:
                weight_loss[:,i,z] = out_data[:,i,z]

    return weight_loss

def Lpde(state_bf, state_af, dt, Re = 0.4, Lx = 243, Ly = 65):
    nx = state_bf.shape[2]
    ny = state_bf.shape[3]
    device = state_af.device
    
    u_bf = state_bf.permute(0,2,3,1)[..., :2]
    p_bf = state_bf.permute(0,2,3,1)[..., -1].reshape(-1, nx, ny, 1)
    u_af = state_af[..., :2]
    
    ux, uy = fdmd2D(u_bf, device, Lx, Ly)
    px, py = fdmd2D(p_bf, device, Lx, Ly)
    uxx, _ = fdmd2D(ux, device, Lx, Ly)
    _, uyy = fdmd2D(uy, device, Lx, Ly)

    u_lap = uxx + uyy
    p_grad = torch.cat((px, py), -1)
    
    L_state = (u_af - u_bf) / dt + u_bf[..., 0].reshape(-1, nx, ny, 1) * ux + \
        u_bf[..., 1].reshape(-1, nx, ny, 1) * uy - Re * u_lap + p_grad 

    loss = (L_state ** 2).mean()

    return L_state

def calMean(data_list):
    ans = []
    for data in data_list:
        length = data.shape[0]
        if (length % 10 != 0):
            data = data[:-(length % 10)]
        data = data.reshape(length // 10, 10, -1).mean(1)
        ans.append(data)
    return ans

def calVar(data_list):
    ans = []
    for data in data_list:
        length = data.shape[0]
        if (length % 10 != 0):
            data = data[:-(length % 10)]
        data_min = data.reshape(length // 10, 10, -1).min(1)
        data_max = data.reshape(length // 10, 10, -1).max(1)
        ans.append([data_min.values, data_max.values])
    return ans


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class PredLog():
    def __init__(self, length):
        self.length = length
        self.loss1 = AverageMeter()
        self.loss2 = AverageMeter()
        self.loss3 = AverageMeter()
        self.loss4 = AverageMeter()
        self.loss5 = AverageMeter()
        self.loss6 = AverageMeter()
        self.loss7 = AverageMeter()
        self.loss8 = AverageMeter()
        self.loss9 = AverageMeter()
    
    def update(self, loss_list):
        for i in range(len(loss_list)):
            exec(f'self.loss{i+1}.update(loss_list[{i}], self.length)')

    def save_log(self, logs):
        logs['test_pred_loss'].append(self.loss1.avg)
        logs['test_input_en_loss'].append(self.loss2.avg)
        logs['test_output_en_loss'].append(self.loss3.avg)
        logs['test_physics_loss'].append(self.loss4.avg)
        logs['test_phys_encode_loss'].append(self.loss5.avg)
        logs['test_hidden_loss'].append(self.loss6.avg)
        logs['test_error_pred'].append(self.loss7.avg)
        logs['test_error_input'].append(self.loss8.avg)
        logs['test_error_output'].append(self.loss9.avg)
    
    def save_train(self, logs):
        logs['train_pred_loss'].append(self.loss1.avg)
        logs['train_input_en_loss'].append(self.loss2.avg)
        logs['train_output_en_loss'].append(self.loss3.avg)
        logs['train_physics_loss'].append(self.loss4.avg)
        logs['train_phys_encode_loss'].append(self.loss5.avg)
        logs['train_hidden_loss'].append(self.loss6.avg)
        logs['train_error_pred'].append(self.loss7.avg)
        logs['train_error_input'].append(self.loss8.avg)
        logs['train_error_output'].append(self.loss9.avg)

    def save_phys_retrain(self, logs):
        logs['retrain_phys_loss'].append(self.loss1.avg)

    def save_data_retrain(self, logs):
        logs['retrain_data_loss'].append(self.loss1.avg)


class LoadData:
    def __init__(self, data_path):
        self.data = torch.load(data_path)
        self.data_set = NSE_Dataset
        self.norm = dict()
        self.Ndata = 0
        self.init_set()

    def init_set(self):
        "to be finished"
        pass 

    def get_params(self):
        "to be finished"
        pass
    
    def get_obs(self):
        return self.obs

    def get_data(self):
        "to be finished"
        pass
    
    def split(self):
        "to be finished"
        pass

    def normalize(self, method = 'unif', logs = None):
        "to be finished"
        pass
    
    def unnormalize(self):
        "to be finished"
        pass
    
    def toGPU(self):
        "to be finished"
        pass
    
    def trans2TrainingSet(self, batch_size, rate):
        NSE_data = self.data_set(self)

        tr_num = int(rate * self.Ndata)
        ts_num = int(0.1 * self.Ndata)
        train_data, test_data, _ = random_split(NSE_data, [tr_num, ts_num, self.Ndata - tr_num - ts_num])

        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, drop_last=True)

        return train_loader, test_loader
    
    def trans2CheckSet(self, batch_size, rate):
        NSE_data = self.data_set(self)
        tr_num = int(rate * self.Ndata)
        check_data, _ = random_split(NSE_data, [tr_num, self.Ndata - tr_num])
        data_loader = DataLoader(dataset=check_data, batch_size=batch_size, shuffle=True, drop_last=True)
        return data_loader


class LoadDataNSE(LoadData):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.data_set = NSE_Dataset

    def init_set(self):
        self.all_data = self.data

        self.get_params()

    def get_data(self):
        return self.data
    
    def get_params_2(self):
        self.nt_ctr = self.ctr.shape[1]
        self.N0_ctr = self.ctr.shape[0]
        self.Ndata_ctr = self.N0_ctr * self.nt_ctr

        self.nt_c = self.Cd.shape[-1]
        self.N0_c = self.Cd.shape[0]
        self.Ndata_c = self.N0_c * self.nt_c
        
        return self.N0_ctr, self.nt_ctr, self.N0_c, self.nt_c, self.nx, self.ny
    
    def get_params(self):
        self.nt = self.all_data.shape[-2]
        self.N0 = self.all_data.shape[-1]

        return self.N0, self.nt
    
    def toGPU(self):
        self.obs = self.obs.cuda()
        self.Cd = self.Cd.cuda()
        self.Cl = self.Cl.cuda()
        self.ctr = self.ctr.cuda()

    def split(self, Ng, tg):
        print(self.ctr.shape[-2])
        self.Ng = Ng
        self.tg = tg
        self.dt = tg * 0.1
        self.obs = self.obs[::Ng, ::tg]
        self.Cd, self.Cl = self.Cd[::Ng, tg-1::tg], self.Cl[::Ng, tg-1::tg]
        self.ctr = self.ctr[::Ng, ::tg] 

        self.get_params()
        return self.obs, self.Cd, self.Cl, self.ctr

    def normalize(self, method = 'unif', logs = None):
        if method == 'unif':
            Cd_min, Cd_range = self.Cd.min(), self.Cd.max() - self.Cd.min()
            Cl_min, Cl_range = self.Cl.min(), self.Cl.max() - self.Cl.min()
            ctr_min, ctr_range = self.ctr.min(), self.ctr.max() - self.ctr.min()
            obs_min, obs_range = self.obs.min(), self.obs.max() - self.obs.min()
            self.Cd = (self.Cd - Cd_min) / Cd_range
            self.Cl = (self.Cl - Cl_min) / Cl_range

            self.norm['Cd'] = [Cd_min, Cd_range]
            self.norm['Cl'] = [Cl_min, Cl_range]
            self.norm['ctr'] = [ctr_min, ctr_range]
            self.norm['obs'] = [obs_min, obs_range]

        elif method == 'logs_unif':
            logs = logs['data_norm']
            Cd_min, Cd_range = logs['Cd']
            Cl_min, Cl_range = logs['Cl']
            ctr_min, ctr_range = logs['ctr']
            obs_min, obs_range = logs['obs']

            self.Cd = (self.Cd - Cd_min) / Cd_range
            self.Cl = (self.Cl - Cl_min) / Cl_range
            self.norm = logs

        return self.norm
    
    def unnormalize(self):
        Cd_min, Cd_range = self.norm['Cd']
        Cl_min, Cl_range = self.norm['Cl']

        self.Cd = self.Cd * Cd_range + Cd_min
        self.Cl = self.Cl * Cl_range + Cl_min


class NSE_Dataset(Dataset):
    def __init__(self, data, mode='grid'):
        if (mode == 'grid'):

            data_run = data.get_data()
            print(data_run.shape)
            input_data =torch.tensor(data_run[:, :-1, :])
            output_data = torch.tensor(data_run[:, -1, :])
            self.Ndata = data.Ndata

            u_i = input_data[0,:]
            v_i = input_data[1,:]
            p_i = input_data[2,:]

            ## normalize
            u_i_mean, u_i_std = u_i.mean(), u_i.std()
            v_i_mean, v_i_std = v_i.mean(), v_i.std()
            p_i_mean, p_i_std = p_i.mean(), p_i.std()

            u_i = (u_i - u_i_mean) / u_i_std
            v_i = (v_i - v_i_mean) / v_i_std
            p_i = (p_i - p_i_mean) / p_i_std

            u_o = output_data[0,:]
            v_o = output_data[1,:]
            p_o = output_data[2,:]

            ## normalize
            u_o_mean, u_o_std = u_o.mean(), u_o.std()
            v_o_mean, v_o_std = v_o.mean(), v_o.std()
            p_o_mean, p_o_std = p_o.mean(), p_o.std()

            u_o = (u_o - u_o_mean) / u_o_std
            v_o = (v_o - v_o_mean) / v_o_std
            p_o = (p_o - p_o_mean) / p_o_std

        self.ipt = torch.cat((u_i, v_i,p_i), dim=-1)
        self.opt = torch.cat((u_o, v_o, p_o), dim=-1)
        
    def __len__(self):
        return self.Ndata

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.ipt[idx])
        y = torch.FloatTensor(self.opt[idx])
        return x, y


class RBC_Dataset(Dataset):
    def __init__(self, data):
        N0, nt, nx, ny = data.get_params()
        obs, temp, ctr = data.get_data()
        self.Ndata = N0 * nt
        ctr = ctr.reshape(-1, nx, ny, 1)
        input_data = obs[:, :-1].reshape(-1, nx, ny, 3)
        output_data = obs[:, 1:].reshape(-1, nx, ny, 3)

        self.ipt = torch.cat((input_data, ctr), dim=-1)
        self.opt = output_data
        
    def __len__(self):
        return self.Ndata

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.ipt[idx])
        y = torch.FloatTensor(self.opt[idx])
        return x, y