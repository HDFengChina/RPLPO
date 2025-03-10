
import os
import numpy as np
import torch
import itertools
from functools import partial
from torch.utils import data
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import scipy
import scipy.io
from scipy.io import loadmat
import sys

class SWE_Linear():
    def __init__(self,
                 xmin=0,
                 xmax=1,
                 ymin=0,
                 ymax=1,
                 Nx=100,
                 Ny=100,
                 g=1.0,# 9.81,
                 f=0,
                 havg=1.0,
                 dt=1.0e-3,
                 tend=1.0,
                 device=None,
                 dtype=torch.float64,                                                  
                 ):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.Nx = Nx
        self.Ny = Ny
        x = torch.linspace(xmin, xmax, Nx + 1, device=device, dtype=dtype)[:-1]
        y = torch.linspace(ymin, ymax, Ny + 1, device=device, dtype=dtype)[:-1]
        self.x = x
        self.y = y
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
        self.g = g
        # self.f = 2*Omega*np.sin(lat)
        self.f = f
        self.havg = havg
        self.h = torch.zeros_like(self.X, device=device)
        self.h0 = torch.zeros_like(self.h, device=device)
        self.u = torch.zeros_like(self.X, device=device)
        self.u0 = torch.zeros_like(self.u, device=device)
        self.v = torch.zeros_like(self.X, device=device)
        self.v0 = torch.zeros_like(self.v, device=device)
        self.dt = dt
        self.tend = tend
        self.t = 0
        self.it = 0
        self.H = []
        self.U = []
        self.V = []
        self.T = []
        self.device = device
    
    def initialize_gaussian(self,amp=1.0, sigma=0.05, loc=[0.5,0.5]):
        loc_x = loc[0]
        loc_y = loc[1]

        h0 = amp*torch.exp(-((self.X-loc_x)**2/(2*(sigma)**2) + (self.Y-loc_y)**2/(2*(sigma)**2)))
        return h0
        
        
    # All Central Differencing Functions are 4th order.  These are used to compute ann inputs.
    def CD_i(self, data, axis, dx):
        data_m2 = torch.roll(data,shifts=2,dims=axis)
        data_m1 = torch.roll(data,shifts=1,dims=axis)
        data_p1 = torch.roll(data,shifts=-1,dims=axis)
        data_p2 = torch.roll(data,shifts=-2,dims=axis)
        data_diff_i = (data_m2 - 8.0*data_m1 + 8.0*data_p1 - data_p2)/(12.0*dx)
        return data_diff_i

    def CD_ij(self, data, axis_i, axis_j, dx, dy):
        data_diff_i = self.CD_i(data,axis_i,dx)
        data_diff_ij = self.CD_i(data_diff_i,axis_j,dy)
        return data_diff_ij

    def CD_ii(self, data, axis, dx):
        data_m2 = torch.roll(data,shifts=2,dims=axis)
        data_m1 = torch.roll(data,shifts=1,dims=axis)
        data_p1 = torch.roll(data,shifts=-1,dims=axis)
        data_p2 = torch.roll(data,shifts=-2,dims=axis)
        data_diff_ii = (-data_m2 + 16.0*data_m1 - 30.0*data + 16.0*data_p1 -data_p2)/(12.0*dx**2)
        return data_diff_ii

    def Dx(self, data):
        data_dx = self.CD_i(data=data, axis=0, dx=self.dx)
        return data_dx
    
    def Dy(self, data):
        data_dy = self.CD_i(data=data, axis=1, dx=self.dy)
        return data_dy

    def Dxx(self, data):
        data_dxx = self.CD_ii(data, axis=0, dx=self.dx)
        return data_dxx

    def Dyy(self, data):
        data_dyy = self.CD_ii(data, axis=1, dx=self.dy)
        return data_dyy

    def calc_RHS(self, h, u, v):
        h_x = self.Dx(h)
        h_y = self.Dy(h)
        u_x = self.Dx(u)
        v_y = self.Dy(v)
        
        h_RHS = -self.havg*(u_x + v_y)
        u_RHS = self.f*v - self.g*h_x
        v_RHS = -self.f*u - self.g*h_y
        return h_RHS, u_RHS, v_RHS
        
    def update_field(self, field, RHS, step_frac):
        field_new = field + self.dt*step_frac*RHS
        return field_new
        

    def rk4_merge_RHS(self, field, RHS1, RHS2, RHS3, RHS4):
        field_new = field + self.dt/6.0*(RHS1 + 2*RHS2 + 2.0*RHS3 + RHS4)
        return field_new

    def rk4(self, h, u, v, t=0):
        h_RHS1, u_RHS1, v_RHS1 = self.calc_RHS(h, u, v)
        t1 = t + 0.5*self.dt
        h1 = self.update_field(h, h_RHS1, step_frac=0.5)
        u1 = self.update_field(u, u_RHS1, step_frac=0.5)
        v1 = self.update_field(v, v_RHS1, step_frac=0.5)
        
        h_RHS2, u_RHS2, v_RHS2 = self.calc_RHS(h1, u1, v1)
        t2 = t + 0.5*self.dt
        h2 = self.update_field(h, h_RHS2, step_frac=0.5)
        u2 = self.update_field(u, u_RHS2, step_frac=0.5)
        v2 = self.update_field(v, v_RHS2, step_frac=0.5)
        
        h_RHS3, u_RHS3, v_RHS3 = self.calc_RHS(h2, u2, v2)
        t3 = t + self.dt
        h3 = self.update_field(h, h_RHS3, step_frac=1.0)
        u3 = self.update_field(u, u_RHS3, step_frac=1.0)
        v3 = self.update_field(v, v_RHS3, step_frac=1.0)
        
        h_RHS4, u_RHS4, v_RHS4 = self.calc_RHS(h3, u3, v3)
        
        t_new = t + self.dt
        h_new = self.rk4_merge_RHS(h, h_RHS1, h_RHS2, h_RHS3, h_RHS4)
        u_new = self.rk4_merge_RHS(u, u_RHS1, u_RHS2, u_RHS3, u_RHS4)
        v_new = self.rk4_merge_RHS(v, v_RHS1, v_RHS2, v_RHS3, v_RHS4)
        
        return h_new, u_new, v_new, t_new
    
    def plot_data(self, cmap='jet', vmin=None, vmax=None, fig_num=0, title='', xlabel='', ylabel=''):
        plt.ion()
        fig = plt.figure(fig_num)
        plt.cla()
        plt.clf()
        
        c = plt.pcolormesh(self.X, self.Y, self.h, cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')
        fig.colorbar(c)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axis('equal')
        plt.axis('square')
        plt.draw() 
        plt.pause(1e-17)
        plt.show()
        
        
    def driver(self, h0, save_interval=10, plot_interval=0):
        self.h0 = h0[:self.Nx,:self.Ny]
        self.h = self.h0
        self.u = self.u0
        self.v = self.v0
        self.t = 0
        self.it = 0
        self.T = []
        self.H = []
        self.U = []
        self.V = []
        
        if plot_interval != 0 and self.it % plot_interval == 0:
            self.plot_data(vmin=-1,vmax=1,title=r'\{u}')
        if save_interval != 0 and self.it % save_interval == 0:
            self.H.append(self.h)
            self.U.append(self.u)
            self.V.append(self.v)
            self.T.append(self.t)
        # Compute equations
        while self.t < self.tend:
            self.h, self.u, self.v, self.t = self.rk4(self.h, self.u, self.v, self.t)
            
            self.it += 1
            if plot_interval != 0 and self.it % plot_interval == 0:
                self.plot_data(vmin=-1,vmax=1,title=r'\{u}')
            if save_interval != 0 and self.it % save_interval == 0:
                self.H.append(self.h)
                self.U.append(self.u)
                self.V.append(self.v)
                self.T.append(self.t)

        return torch.stack(self.H), torch.stack(self.U), torch.stack(self.V)
    
class SWE_Nonlinear():
    def __init__(self,
                 xmin=0,
                 xmax=1,
                 ymin=0,
                 ymax=1,
                 Nx=100,
                 Ny=100,
                 g=1.0,
                 nu=0.001,
                 dt=1.0e-2,
                 tend=1.0,
                 device=None,
                 dtype=torch.float64,                                                  
                 ):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.Nx = Nx
        self.Ny = Ny
        x = torch.linspace(xmin, xmax, Nx + 1, device=device, dtype=dtype)[:-1]
        y = torch.linspace(ymin, ymax, Ny + 1, device=device, dtype=dtype)[:-1]
        self.x = x
        self.y = y
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
        self.g = g
        self.nu = nu
        self.h = torch.zeros_like(self.X, device=device)
        self.h0 = torch.zeros_like(self.h, device=device)
        self.u = torch.zeros_like(self.X, device=device)
        self.u0 = torch.zeros_like(self.u, device=device)
        self.v = torch.zeros_like(self.X, device=device)
        self.v0 = torch.zeros_like(self.v, device=device)
        self.dt = dt
        self.tend = tend
        self.t = 0
        self.it = 0
        self.H = []
        self.U = []
        self.V = []
        self.T = []
        self.device = device
        self.earth_para = 1.46e-4
        
    
    def initialize_gaussian(self,amp=0.1, sigma=0.05, loc=[0.5,0.5]):
        loc_x = loc[0]
        loc_y = loc[1]

        # There are three conserved quantities - initialize
        h0 = 1.0 + amp*torch.exp(-((self.X-loc_x)**2/(2*(sigma)**2) + (self.Y-loc_y)**2/(2*(sigma)**2)))
        return h0
        
        
    # All Central Differencing Functions are 4th order.  These are used to compute ann inputs.
    def CD_i(self, data, axis, dx):
        data_m2 = torch.roll(data,shifts=2,dims=axis)
        data_m1 = torch.roll(data,shifts=1,dims=axis)
        data_p1 = torch.roll(data,shifts=-1,dims=axis)
        data_p2 = torch.roll(data,shifts=-2,dims=axis)
        data_diff_i = (data_m2 - 8.0*data_m1 + 8.0*data_p1 - data_p2)/(12.0*dx)
        return data_diff_i

    def CD_ij(self, data, axis_i, axis_j, dx, dy):
        data_diff_i = self.CD_i(data,axis_i,dx)
        data_diff_ij = self.CD_i(data_diff_i,axis_j,dy)
        return data_diff_ij

    def CD_ii(self, data, axis, dx):
        data_m2 = torch.roll(data,shifts=2,dims=axis)
        data_m1 = torch.roll(data,shifts=1,dims=axis)
        data_p1 = torch.roll(data,shifts=-1,dims=axis)
        data_p2 = torch.roll(data,shifts=-2,dims=axis)
        data_diff_ii = (-data_m2 + 16.0*data_m1 - 30.0*data + 16.0*data_p1 -data_p2)/(12.0*dx**2)
        return data_diff_ii

    def Dx(self, data):
        data_dx = self.CD_i(data=data, axis=0, dx=self.dx)
        return data_dx
    
    def Dy(self, data):
        data_dy = self.CD_i(data=data, axis=1, dx=self.dy)
        return data_dy

    def Dxx(self, data):
        data_dxx = self.CD_ii(data, axis=0, dx=self.dx)
        return data_dxx

    def Dyy(self, data):
        data_dyy = self.CD_ii(data, axis=1, dx=self.dy)
        return data_dyy

    def calc_RHS(self, h, u, v, z_bottom):
        
        h_flux_x = self.Dx(h*u)
        h_flux_y = self.Dy(h*v)
        u_flux_x = self.Dx(h*u**2 + 0.5*self.g*h**2)
        u_flux_y = self.Dy(h*u*v)
        u_xx = self.Dxx(u)
        u_yy = self.Dyy(u)
        u_visc = self.nu*(u_xx + u_yy)
        v_flux_x = self.Dx(h*u*v)
        v_flux_y = self.Dy(h*v**2 + 0.5*self.g*h**2)
        v_xx = self.Dxx(v)
        v_yy = self.Dyy(v)
        v_visc = self.nu*(v_xx + v_yy)
        
        z_flux_x = self.g*h*self.Dx(z_bottom)
        z_flux_y = self.g*h*self.Dy(z_bottom)
        
        h_RHS = -(h_flux_x + h_flux_y)
        u_RHS = -(u_flux_x + u_flux_y) + z_flux_x - self.earth_para*v + u_visc
        v_RHS = -(v_flux_x + v_flux_y) + z_flux_y + self.earth_para*u + v_visc
        return h_RHS, u_RHS, v_RHS
        
    def update_field(self, field, RHS, step_frac):
        field_new = field + self.dt*step_frac*RHS
        return field_new
        

    def rk4_merge_RHS(self, field, RHS1, RHS2, RHS3, RHS4):
        field_new = field + self.dt/6.0*(RHS1 + 2*RHS2 + 2.0*RHS3 + RHS4)
        return field_new

    def rk4(self, h, u, v, z_bottom, t=0):
        
        h_RHS1, u_RHS1, v_RHS1 = self.calc_RHS(h, u, v, z_bottom)

        t1 = t + 0.5*self.dt
        h1 = self.update_field(h, h_RHS1, step_frac=0.5)
        u1 = self.update_field(u, u_RHS1, step_frac=0.5)
        v1 = self.update_field(v, v_RHS1, step_frac=0.5)
        
        h_RHS2, u_RHS2, v_RHS2 = self.calc_RHS(h1, u1, v1, z_bottom)

        t2 = t + 0.5*self.dt
        h2 = self.update_field(h, h_RHS2, step_frac=0.5)
        u2 = self.update_field(u, u_RHS2, step_frac=0.5)
        v2 = self.update_field(v, v_RHS2, step_frac=0.5)
        
        h_RHS3, u_RHS3, v_RHS3 = self.calc_RHS(h2, u2, v2, z_bottom)
        t3 = t + self.dt
        h3 = self.update_field(h, h_RHS3, step_frac=1.0)
        u3 = self.update_field(u, u_RHS3, step_frac=1.0)
        v3 = self.update_field(v, v_RHS3, step_frac=1.0)
        
        h_RHS4, u_RHS4, v_RHS4 = self.calc_RHS(h3, u3, v3, z_bottom)
        
        t_new = t + self.dt
        h_new = self.rk4_merge_RHS(h, h_RHS1, h_RHS2, h_RHS3, h_RHS4)
        u_new = self.rk4_merge_RHS(u, u_RHS1, u_RHS2, u_RHS3, u_RHS4)
        v_new = self.rk4_merge_RHS(v, v_RHS1, v_RHS2, v_RHS3, v_RHS4)
        
        return h_new, u_new, v_new, t_new
    
    def rk2(self, h, u, v, z_bottom, t=0):
        h_RHS1, u_RHS1, v_RHS1 = self.calc_RHS(h, u, v, z_bottom)
        
        t1 = t + 0.5*self.dt
        h1 = self.update_field(h, h_RHS1, step_frac=0.5)
        u1 = self.update_field(u, u_RHS1, step_frac=0.5)
        v1 = self.update_field(v, v_RHS1, step_frac=0.5)
        
        h_RHS2, u_RHS2, v_RHS2 = self.calc_RHS(h1, u1, v1, z_bottom)

        t_new = t + self.dt
        h_new = h + 0.5*self.dt*(h_RHS1 + h_RHS2)
        u_new = u + 0.5*self.dt*(u_RHS1 + u_RHS2)
        v_new = v + 0.5*self.dt*(v_RHS1 + v_RHS2)
        
        return h_new, u_new, v_new, t_new
    
    def plot_data(self, cmap='jet', vmin=None, vmax=None, fig_num=0, title='', xlabel='', ylabel=''):
        plt.ion()
        fig = plt.figure(fig_num)
        plt.cla()
        plt.clf()
        
        c = plt.pcolormesh(self.X, self.Y, self.h, cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')
        fig.colorbar(c)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axis('equal')
        plt.axis('square')
        plt.draw() 
        plt.pause(1e-17)
        plt.show()
        
        
    def driver(self, h0, z_bottom, save_interval=1, plot_interval=0):

        self.h0 = h0[:self.Nx,:self.Ny]
        self.z_bottom = z_bottom[:self.Nx,:self.Ny]

        self.h = self.h0
        self.u = self.u0
        self.v = self.v0
        self.t = 0
        self.it = 0
        self.T = []
        self.H = []
        self.U = []
        self.V = []
        
        if plot_interval != 0 and self.it % plot_interval == 0:
            self.plot_data(vmin=-1,vmax=1,title=r'\{u}')
        if save_interval != 0 and self.it % save_interval == 0:
            self.H.append(self.h)
            self.U.append(self.u)
            self.V.append(self.v)
            self.T.append(self.t)
        # Compute equations
        while self.t < self.tend:
            self.h, self.u, self.v, self.t = self.rk4(self.h, self.u, self.v, self.z_bottom, self.t)
            
            self.it += 1
            if plot_interval != 0 and self.it % plot_interval == 0:
                self.plot_data(vmin=-1,vmax=1,title=r'\{u}')
            if save_interval != 0 and self.it % save_interval == 0:
                self.H.append(self.h)
                self.U.append(self.u)
                self.V.append(self.v)
                self.T.append(self.t)

        return torch.stack(self.H), torch.stack(self.U), torch.stack(self.V)