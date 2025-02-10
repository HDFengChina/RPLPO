import torch
from script.swe import SWE_Nonlinear
import yaml

class PDE_loss():
    def __init__(self, device):
        super(PDE_loss, self).__init__()
        z_path = '/users/swe_data/swe_bottom_3'
        self.z_bottom = torch.load(z_path, map_location=device)
        config_file = '/users/swe_code/swe/script/swe_nonlinear.yaml'  
        config = self.load_config(config_file)
        Nsamples = 1
        N = config['data']['nx']
        Nt = config['data']['nt']
        nu = config['data']['nu']
        g = config['data']['g']
        H = config['data']['H']
        Nx = N
        Ny = N
        dim = 2
        l = 0.1
        L = 1.0
        sigma = 1 
        Nu = None 
        dt = 1.0e-2
        tend = 1.0
        self.swe_eq = SWE_Nonlinear(Nx=Nx, Ny=Ny, g=g, nu=nu, dt=dt, tend=tend, device=device)

    def load_config(self, file):
        with open(file, 'r') as f:
            config = yaml.load(f, yaml.FullLoader)
        return config

    def physics_loss(self, input, output):

        u1 = input[:,:,:,0].permute(1,2,0)
        v1 = input[:,:,:,1].permute(1,2,0)
        h1 = input[:,:,:,2].permute(1,2,0)
        u2 = output[:,:,:,0].permute(1,2,0)  # output
        v2 = output[:,:,:,1].permute(1,2,0)  # output
        h2 = output[:,:,:,2].permute(1,2,0)  # output

        h_label, u_label, v_label, t11 = self.swe_eq.rk4(h1, u1, v1, self.z_bottom.permute(1,2,0).repeat(1,1,32), 0)
        loss_u = ((u_label - u2)**2).mean()
        loss_v = ((v_label - v2)**2).mean()
        loss_h = ((h_label - h2)**2).mean()

        return loss_u+loss_v+loss_h

    



