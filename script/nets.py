import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from script.utils import *
from script.unet import *

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1    # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixt,ioxt->boxt", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)  # 二维傅里叶变换

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device) # shape(16,48,243,33)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class encoder_en(nn.Module):
    def __init__(self, out_channels, recall_size, width=5):
        super(encoder_en, self).__init__()
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(3*recall_size, 64, width, padding=2),
            nn.Tanh(),
            nn.Conv2d(64, 32, width, padding=2),
            nn.Tanh(),
            nn.Conv2d(32, 16, width, padding=2),
            nn.Tanh(),
            nn.Conv2d(16, out_channels, width, padding=2),
        )

    def forward(self, input):
        # [batch_size] ——> [batch_size, nx, ny, 1]
        input = self.net(input)
        return input    # [batch_size, out_channels, nx, ny]
    

class encoder_de():
    def __init__(self, sample_gap):
        super(encoder_de, self).__init__()
        self.sample_gap = sample_gap
                
    def output(self, out_data):
        self.out_data = out_data
        self.sample = torch.tensor(np.zeros((self.out_data.shape[0], self.out_data.shape[1], int((self.out_data.shape[-2]-1)/self.sample_gap)+1, int((self.out_data.shape[-1]-1)/self.sample_gap)+1)), dtype=torch.float64).to(self.out_data.device)
        loop_l = 0
        for i in range(self.out_data.shape[-2]):
            loop_c = 0
            if i%self.sample_gap == 0:
                for z in range(self.out_data.shape[-1]):
                    if z%self.sample_gap == 0:
                        self.sample[:,:,loop_l,loop_c] = self.out_data[:,:,i,z]
                        self.sample[:,:,loop_l,loop_c] = self.out_data[:,:,i,z]
                        self.sample[:,:,loop_l,loop_c] = self.out_data[:,:,i,z]
                        loop_c += 1
                loop_l +=1
        
        return self.sample
    
class state_en(nn.Module):
    def __init__(self, modes1, modes2, width, L, Lx, Ly):
        super(state_en, self).__init__()
        self.Lx = Lx
        self.Ly = Ly
        self.fc0 = nn.Linear(5, width)
        self.down = [ FNO_layer(modes1, modes2, width) for i in range(L-1) ]
        self.down += [ FNO_layer(modes1, modes2, width, last=True) ]
        self.down = nn.Sequential(*self.down)

    def forward(self, x):
        x = x.permute(0,2,3,1)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)    # [batch_size, nx, ny, 5]
        x = self.fc0(x)  
        x = x.permute(0, 3, 1, 2)
        x_latent = self.down(x) 
        return x_latent     # [batch_size, width, nx, ny]
    
    def get_grid(self, shape, device):
        batchsize, nx, ny = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, self.Lx, nx), dtype=torch.float)
        gridx = gridx.reshape(1, nx, 1, 1).repeat([batchsize, 1, ny, 1])
        gridy = torch.tensor(np.linspace(0, self.Ly, ny), dtype=torch.float)
        gridy = gridy.reshape(1, 1, ny, 1).repeat([batchsize, nx, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class state_de(nn.Module):
    def __init__(self, modes1, modes2, width, L):
        super(state_de, self).__init__()

        self.up = [ FNO_layer(modes1, modes2, width) for i in range(L-1) ]
        self.up += [ FNO_layer(modes1, modes2, width, last=True) ]
        self.up = nn.Sequential(*self.up)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x_latent):
        x = self.up(x_latent)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x    # [batch_size, nx, ny, 5]
    
class FNO_layer(nn.Module):
    def __init__(self, modes1, modes2, width, extra_channels=0, last=False):
        super(FNO_layer, self).__init__()
        """ ...
        """
        self.last = last

        width = width + extra_channels
        self.conv = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)

    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x, dim_t)"""

        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        if not self.last:
            x = F.gelu(x)
                        
        return x
        

class MyDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor

        self.target_tensor = target_tensor

    def __len__(self):
        return self.data_tensor.size(0)
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]


def data_repeat(data, num_obs, num_state, gap):
    
    data_run = data

    num = data_run.shape[1]
    data_run = data_run

    u_sum = torch.zeros((data_run.shape[2],num,num_state,num_state), dtype=torch.float64)
    v_sum = torch.zeros((data_run.shape[2],num,num_state,num_state), dtype=torch.float64)
    h_sum = torch.zeros((data_run.shape[2],num,num_state,num_state), dtype=torch.float64)

    for uvh in range(data_run.shape[0]):  # uh:2
        for t in range(data_run.shape[2]):  # t:1001

            u_m = torch.zeros((num,num_obs,num_obs), dtype=torch.float64)
            

            for i in range(u_m.shape[1]):
                u_m[:,i] = data_run[uvh,:,t,i*u_m.shape[1]:i*u_m.shape[1] + u_m.shape[2]]

            u_s = torch.zeros((num,num_state,num_state), dtype=torch.float64)

            for i in range(u_s.shape[1]):
                if i%gap == 0:
                    for z in range(u_s.shape[2]):
                        if z%gap == 0:
                            u_s[:,i,z] = u_m[:, int(i/gap), int(z/gap)]
            
            if uvh == 0:
                u_sum[t] = u_s
            elif uvh == 1:
                v_sum[t] = u_s
            elif uvh == 2:
                h_sum[t] = u_s

    sum = torch.stack((u_sum, v_sum, h_sum), 0)
    return sum


def data_arrange(data_path):
    data_run = torch.load(data_path)
    u_sum = []
    v_sum = []
    p_sum = []
    for uvp in range(len(data_run)):  # uvp:3
        for t in range(len(data_run[uvp])):  # t:8000

            u_m = np.zeros((13,13))

            for i in range(u_m.shape[0]):
                u_m[i] = data_run[uvp][t][i*u_m.shape[1]:i*u_m.shape[1] + u_m.shape[1]]
            
            if uvp == 0:
                u_sum.append(u_m)
            elif uvp == 1:
                v_sum.append(u_m)
            elif uvp == 2:
                p_sum.append(u_m)

    sum = [u_sum, v_sum, p_sum]
    return sum

class control_en(nn.Module):
    def __init__(self, out_channels, width=5):
        super(control_en, self).__init__()
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, width, padding=2),
            nn.Tanh(),
            nn.Conv2d(64, 32, width, padding=2),
            nn.Tanh(),
            nn.Conv2d(32, 16, width, padding=2),
            nn.Tanh(),
            nn.Conv2d(16, out_channels, width, padding=2),
        )

    def forward(self, ctr):
        # [batch_size] ——> [batch_size, nx, ny, 1]
        ctr = ctr.permute(0, 3, 1, 2)
        ctr = self.net(ctr)
        return ctr    # [batch_size, out_channels, nx, ny]
    

class control_de(nn.Module):
    def __init__(self, in_channels, width=5):
        super(control_de, self).__init__()
        self.in_channels = in_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, width, padding=2),
            nn.Tanh(),
            nn.Conv2d(64, 32, width, padding=2),
            nn.Tanh(),
            nn.Conv2d(32, 16, width, padding=2),
            nn.Tanh(),
            nn.Conv2d(16, 1, width, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2)
        )

        self.out = nn.Sequential(
            nn.Linear(400, 3*32),
            nn.ReLU(),
            nn.Linear(3*32, 1)
        ) 

    def forward(self, ctr):
        ctr = self.net(ctr)
        ctr = ctr.reshape(ctr.size(0), -1)
        ctr =self.out(ctr)
        return ctr    # [batch_size]
    
class FNO_layer_trans(nn.Module):
    def __init__(self, modes1, modes2, width, extra_channels=0, last=False):
        super(FNO_layer_trans, self).__init__()
        """ ...
        """
        self.last = last

        self.conv = SpectralConv2d(width+extra_channels, width, modes1, modes2)
        self.w = nn.Conv2d(width+extra_channels, width, 1)

    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x, dim_t)"""

        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        if not self.last:
            x = F.gelu(x)
            
        return x
    
class trans_net(nn.Module):
    def __init__(self, modes1, modes2, width, L, f_channels):
        super(trans_net, self).__init__()
        self.fc0 = nn.Linear(3, width)

        self.trans = [ FNO_layer_trans(modes1, modes2, width, f_channels) ]
        self.trans += [ FNO_layer(modes1, modes2, width) for i in range(L-2) ]
        self.trans += [ FNO_layer(modes1, modes2, width, last=True) ]
        self.trans = nn.Sequential(*self.trans)
        
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 3)
        self.padding = 9

    def forward(self, x, ctr):
        x = x.permute(0,2,3,1)  # 16,41,41,3
        x = self.fc0(x) 
                
        x_latent = x.permute(0, 3, 1, 2)  # 16,32,41,41
        x_latent = F.pad(x_latent, [0,self.padding, 0,self.padding])
        
        trans_out = self.trans(x_latent)
        trans_out = trans_out[..., :-self.padding, :-self.padding]
        trans_out = trans_out.permute(0,2,3,1)
        trans_decode = self.fc1(trans_out)
        
        trans_decode = F.gelu(trans_decode)
        predict = self.fc2(trans_decode)
                
        return trans_out, predict

class set_bound():
    def __init__(self):
        super(set_bound, self).__init__()

    def set_b(self, in_data, out_data):

        out_data[:,0,:,0] = in_data[:,0,0,:]  # u
        out_data[:,0,:,-1] = in_data[:,0,0,:]
        out_data[:,0,0,:] = in_data[:,0,0,:]
        out_data[:,0,-1,:] = in_data[:,0,-1,:]

        out_data[:,1,:,0] = in_data[:,1,0,:]  # v
        out_data[:,1,:,-1] = in_data[:,1,0,:]
        out_data[:,1,0,:] = in_data[:,1,0,:]
        out_data[:,1,-1,:] = in_data[:,1,0,:]

        out_data[:,2,:,0] = out_data[:,2,:,1]  # p
        out_data[:,2,:,-1] = out_data[:,2,:,-2]
        out_data[:,2,0,:] = out_data[:,2,1,:]
        out_data[:,2,-1,:] = in_data[:,2,-1,:]

        return out_data

    def set_delta_b(self, out_delta):
        out_delta[:,:,:,0] = 0
        out_delta[:,:,:,-1] = 0
        out_delta[:,:,0,:] = 0
        out_delta[:,:,-1,:] = 0
        return out_delta
    
class phys_inform_net(nn.Module):
    def __init__(self, params):
        super(phys_inform_net, self).__init__()

        modes1 = params['modes']
        modes2 = params['modes']
        width = params['width']
        L = params['L']
        shape = params['shape']
        f_channels = params['f_channels']
        Lx, Ly = params['Lxy']
        out_channels = params['en_dim']
        gap = params['gap_sample']
        recall_size = params['recall_size']
        dropout = params['dropout']
        channel = params['channel']
        ups = params['ups']
        
        self.model_en = UNet(ch=channel, in_ch=recall_size*3, ch_mult=[1, 2, 2, 2], num_res_blocks=1, dropout=dropout, sample_gap=gap, ups = ups)  # attn=[False,False,False,False],
        self.trans = trans_net(modes1, modes2, width, L, f_channels)  # FNO
        self.model_de = encoder_de(gap)
    
    def forward(self, x, ctr):

        device = x.device  # bz, uvp*n, ox, oy
        x_en = self.model_en(x)  # bz, uvp, x, y
        x_de = self.model_de.output(x_en)  # bz, uvp, ox, oy
        
        trans_out, pred_delta = self.trans(x_en.to(str(device)).to(torch.float32), ctr)  # bz, x, y, uvp
        
        pred_delta = pred_delta.permute(0, 3, 1, 2)  # bz, uvp, x, y
        
        pred = pred_delta + x_en.to(str(device))

        pred_de = self.model_de.output(pred)  # bz, uvp, ox, oy
        
        return pred_de, x_de, pred, x_en
    

class phys_prediction(nn.Module):
    # def __init__(self, params):
    def __init__(self, params):
        super(phys_prediction, self).__init__()

        modes1 = params['modes']
        modes2 = params['modes']
        width = params['width']
        L = params['L'] + 2
        self.Lx, self.Ly = params['Lxy']
        out_channels = params['en_dim']
        recall_size = params['recall_size']
        
        self.model_en = encoder_en(out_channels, recall_size)
        self.net = [ FNO_layer(modes1, modes2, width) for i in range(L-1) ]
        self.net += [ FNO_layer(modes1, modes2, width, last=True) ]
        self.net = nn.Sequential(*self.net)
        self.fc0 = nn.Linear(15, width)  
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x_in, ctr, x_innext, stage):
        x = self.model_en(x_in)
        
        if stage == 1:
            x_next = self.model_en(x_innext)
        elif stage == 2:
            x_next = x_innext
            
        x= x.permute(0, 2, 3, 1)
        x_next= x_next.permute(0, 2, 3, 1)
        grid = self.get_grid(x.shape, x.device) # 2
        u_bf = x[..., :-1]   # 2 ut
        p_bf = x[..., -1].reshape(-1, x.shape[1], x.shape[2], 1) # 1 pt
        u_af = x_next[..., :-1]  # 2 ut+1

        ux, uy = fdmd2D(u_bf, x.device, self.Lx, self.Ly)   # input 2 + 2 delta ut
        px, py = fdmd2D(p_bf, x.device, self.Lx, self.Ly)   # delta napla pt
        uxx, _ = fdmd2D(ux, x.device, self.Lx, self.Ly)
        _, uyy = fdmd2D(uy, x.device, self.Lx, self.Ly) 
        u_lap = uxx + uyy   # input 2 napla ut
        p_grad = torch.cat((px, py), -1)    # input 2
        ipt = torch.cat((grid, u_bf, ctr, u_af, ux, uy, p_grad, u_lap), -1)

        opt = self.fc0(ipt).permute(0, 3, 1, 2)
        opt = self.net(opt).permute(0, 2, 3, 1)
        opt = self.fc1(opt)
        opt = F.gelu(opt)
        opt = self.fc2(opt)

        return opt, x, x_next

    def get_grid(self, shape, device):
        batchsize, nx, ny = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, self.Lx, nx), dtype=torch.float)
        gridx = gridx.reshape(1, nx, 1, 1).repeat([batchsize, 1, ny, 1])
        gridy = torch.tensor(np.linspace(0, self.Ly, ny), dtype=torch.float)
        gridy = gridy.reshape(1, 1, ny, 1).repeat([batchsize, nx, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    

def Lpde(state_bf, state_af, dt, Re = 0.1, Lx = 2.05, Ly = 2.05):
    nx = state_bf.shape[1]
    ny = state_bf.shape[2]
    device = state_af.device

    u_bf = state_bf[..., :2]
    p_bf = state_bf[..., -1].reshape(-1, nx, ny, 1)
    u_af = state_af[..., :2]

    ux, uy = fdmd2D(u_bf, device, Lx, Ly)
    px, py = fdmd2D(p_bf, device, Lx, Ly)
    uxx, _ = fdmd2D(ux, device, Lx, Ly)
    _, uyy = fdmd2D(uy, device, Lx, Ly)

    u_lap = uxx + uyy
    p_grad = torch.cat((px, py), -1)
    L_state = (u_af - u_bf) / dt + u_bf[..., 0].reshape(-1, nx, ny, 1) * ux + \
        u_bf[..., 1].reshape(-1, nx, ny, 1) * uy - Re * u_lap + p_grad # Re 改为0.025

    loss = (L_state ** 2).mean()

    return L_state

def Lpde2(state_bf, state_af, dt, Re = 0.1, Lx = 2.05, Ly = 2.05):
    
    ny = state_bf.shape[1]
    nx = state_bf.shape[2]
    dy = Ly / (ny-1)
    dx = Lx / (nx-1)
    device = state_af.device

    u_bf = state_bf[..., :2]  # (16,41,41,2), t
    p_bf = state_bf[..., -1].reshape(-1,ny,nx,1)
    u_af = state_af[..., :2]  # t+1
    
    loss_u = -u_af[:, 1:-1, 1:-1, 0] + (u_bf[:, 1:-1, 1:-1, 0]-
                         u_bf[:, 1:-1, 1:-1, 0] * dt / dx *
                        (u_bf[:, 1:-1, 1:-1, 0] - u_bf[:, 1:-1, 0:-2, 0]) -
                         u_bf[:, 1:-1, 1:-1, 1] * dt / dy *
                        (u_bf[:, 1:-1, 1:-1, 0] - u_bf[:, 0:-2, 1:-1, 0]) -
                         dt / (2 * 1 * dx) * (p_bf[:, 1:-1, 2:, 0] - p_bf[:, 1:-1, 0:-2, 0]) +
                         Re * (dt / dx**2 *
                        (u_bf[:, 1:-1, 2:, 0] - 2 * u_bf[:, 1:-1, 1:-1, 0] + u_bf[:, 1:-1, 0:-2, 0]) +
                         dt / dy**2 *
                        (u_bf[:, 2:, 1:-1, 0] - 2 * u_bf[:, 1:-1, 1:-1, 0] + u_bf[:, 0:-2, 1:-1, 0])))
    
    loss_v = -u_af[:, 1:-1,1:-1, 1] + (u_bf[:, 1:-1, 1:-1, 1] -
                        u_bf[:, 1:-1, 1:-1, 0] * dt / dx *
                       (u_bf[:, 1:-1, 1:-1, 1] - u_bf[:, 1:-1, 0:-2, 1]) -
                        u_bf[:, 1:-1, 1:-1, 1] * dt / dy *
                       (u_bf[:, 1:-1, 1:-1, 1] - u_bf[:, 0:-2, 1:-1, 1]) -
                        dt / (2 * 1 * dy) * (p_bf[:, 2:, 1:-1, 0] - p_bf[:, 0:-2, 1:-1, 0]) +
                        Re * (dt / dx**2 *
                       (u_bf[:, 1:-1, 2:, 1] - 2 * u_bf[:, 1:-1, 1:-1, 1] + u_bf[:, 1:-1, 0:-2, 1]) +
                        dt / dy**2 *
                       (u_bf[:, 2:, 1:-1, 1] - 2 * u_bf[:, 1:-1, 1:-1, 1] + u_bf[:, 0:-2, 1:-1, 1])))
        
    b             = (1 * (1 / dt * 
                    ((u_bf[:, 1:-1, 2:, 0] - u_bf[:, 1:-1, 0:-2, 0]) / 
                     (2 * dx) + (u_bf[:, 2:, 1:-1, 1] - u_bf[:, 0:-2, 1:-1, 1]) / (2 * dy)) -
                    ((u_bf[:, 1:-1, 2:, 0] - u_bf[:, 1:-1, 0:-2, 0]) / (2 * dx))**2 -
                      2 * ((u_bf[:, 2:, 1:-1, 0] - u_bf[:, 0:-2, 1:-1, 0]) / (2 * dy) *
                           (u_bf[:, 1:-1, 2:, 1] - u_bf[:, 1:-1, 0:-2, 1]) / (2 * dx))-
                          ((u_bf[:, 2:, 1:-1, 1] - u_bf[:, 0:-2, 1:-1, 1]) / (2 * dy))**2))
    
    loss_p = -p_bf[:, 1:-1, 1:-1, 0] + (((p_bf[:, 1:-1, 2:, 0] + p_bf[:, 1:-1, 0:-2, 0]) * dy**2 + 
                          (p_bf[:, 2:, 1:-1, 0] + p_bf[:, 0:-2, 1:-1, 0]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                          b)
    
    
    loss_bound = ((u_af[:,0,:] - 0)**2).mean() + ((u_af[:,:-1,0] - 0)**2).mean() + ((u_af[:,:-1,-1] - 0)**2).mean() \
        + ((u_af[:,-1,:,0] - 1)**2).mean() + ((u_af[:,-1,:,1] - 0)**2).mean() \
        + ((p_bf[:,:,-1,0] - p_bf[:,:,-2,0])**2).mean() + ((p_bf[:,0,:,0] - p_bf[:,1,:,0])**2).mean() \
            + ((p_bf[:,:,0,0] - p_bf[:,:,1,0])**2).mean() + ((p_bf[:,-1,:,0] - 0)**2).mean() \
            + ((u_bf[:,0,:] - 0)**2).mean() + ((u_bf[:,:-1,0] - 0) + (u_bf[:,:-1,-1] - 0)**2).mean()\
                + ((u_bf[:,-1,:,0] - 1)**2).mean() + ((u_bf[:,-1,:,1] - 0)**2).mean()

    return loss_u, loss_v, loss_p, loss_bound