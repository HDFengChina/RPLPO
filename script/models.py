from email.policy import default
import torch
from torch.utils.data import DataLoader
from timeit import default_timer
import copy
from script.nets import *
from script.utils import *
from script.PDE_loss import *

class NSEModel():
    def __init__(self, shape, dt, args):
        print(f'dt: {dt}')
        self.shape = shape
        self.nx, self.ny = self.shape
        self.dt = dt
        self.params = args
        self.device = torch.device('cuda:{}'.format(self.params.gpu) if torch.cuda.is_available() else 'cpu')
        self.pde = PDE_loss(self.device)
        self.t1 = []
        self.t2 = []
        self.t3 = []

    def set_model(self, phys_inform_model = phys_inform_net):
        model_params = dict()
        model_params['modes'] = self.params.modes
        model_params['width'] = self.params.width
        model_params['L'] = self.params.L
        model_params['shape'] = self.shape
        model_params['f_channels'] = self.params.f_channels
        model_params['Lxy'] = [self.Lx, self.Ly]
        model_params['en_dim'] = self.params.en_dim
        model_params['gap_sample'] = self.params.gap_sample
        model_params['recall_size'] = self.params.recall_size
        model_params['dropout'] = self.params.drop
        model_params['channel'] = self.params.channel
        model_params['ups'] = self.params.ups
        
        self.phys_inform_model = phys_inform_model(model_params).to(self.device)
        self.phys_inform_optimizer = torch.optim.Adam(self.phys_inform_model.parameters(), lr=self.params.lr, weight_decay=self.params.wd)  # 
        self.phys_inform_scheduler = torch.optim.lr_scheduler.StepLR(self.phys_inform_optimizer, step_size=self.params.step_size, gamma=self.params.gamma)
    
    def count_params(self):
        c = 0
        for p in list(self.phys_inform_model.parameters()):
            c += reduce(operator.mul, list(p.size())) 
        return c
    
    def save_log(self, logs):
        with torch.no_grad():
            logs['phys_inform_model'].append(copy.deepcopy(self.phys_inform_model.state_dict()))

    def toCPU(self):
        self.phys_inform_model.to('cpu')

            
    def phys_inform(self, epoch, train_dataloader, ctr_train, logs):
        self.phys_inform_model.train()
        args = logs['args']
        recall = args.recall_size
        t1 = default_timer()
        train_log = PredLog(length=self.params.batch_size)

        for x_train, y_train in train_dataloader:
            
            
            self.phys_inform_model.zero_grad()

            x_train, y_train = x_train.to(self.device), y_train.to(self.device)  # bz, uvp*n*2, x, y
            ctr_train = ctr_train.to(self.device)  # bz, x, y, 1
                
            in_train = x_train[:,:3]
            out_train = y_train[:,:3]
            full_in = x_train[:,3:6]
            full_out = y_train[:,3:6]
            for i in range(recall-1):
                in_train = torch.cat([in_train, x_train[:,i*6+6:i*6+9]], dim=1)   # bz, uvp*n, x, y
                out_train = torch.cat([out_train, y_train[:,i*6+6:i*6+9]], dim=1)
                full_in = torch.cat([full_in, x_train[:,i*6+9:i*6+12]], dim=1)
                full_out = torch.cat([full_out, y_train[:,i*6+9:i*6+12]], dim=1)
            
            mean = logs['mean']
            std = logs['std']

            loss1, loss2, loss3, loss5, loss6, trans_out, input_en, error_pred, error_input, error_output = self.pred_loss(in_train, ctr_train, out_train, full_in, full_out, mean, std)
            pde_loss_tr = self.pde.physics_loss(input_en.to(self.device),trans_out.to(self.device))

            loss4 = pde_loss_tr.to(torch.float32)
                
            self.train_step(loss1.to(self.device), loss2.to(self.device), loss3.to(self.device), loss4.to(self.device), loss5.to(self.device), loss6.to(self.device), logs)
            train_log.update([loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.mean().item(),
                            error_pred.item(), error_input.item(), error_output.item()])
                
        train_log.save_train(logs)

        self.scheduler_step()
        t2 = default_timer()
        

        print('# {} train: {:1.2f} | pred: {:1.2e}  input_en: {:1.2e}  output_en: {:1.2e} physics: {:1.2e} phys_encode: {:1.2e} hidden: {:1.2e} error_pred: {:1.2e} error_in: {:1.2e} error_out: {:1.2e}'
              .format(epoch, t2-t1, train_log.loss1.avg, train_log.loss2.avg, train_log.loss3.avg, train_log.loss4.avg, train_log.loss5.avg, train_log.loss6.avg,
                      train_log.loss7.avg, train_log.loss8.avg, train_log.loss9.avg))
        
    def pred_loss(self, ipt, ctr, opt, full_in, full_out, mean, std):

        input = self.phys_inform_model.model_de.output(ipt).to(self.device).to(torch.float32)  # form the input data with downsample      bz, uvp*n, ox, oy
        output = self.phys_inform_model.model_de.output(opt).to(self.device).to(torch.float32)  # form the output data with downsample

        out_en = self.phys_inform_model.model_en(output)  # bz, uvp, x, y

        out_latent = self.phys_inform_model.model_de.output(out_en).to(self.device)  # label of out encode, compare with out_decode     bz, uvp, ox, oy
        
        out_pred, out_de, trans_out, input_en = self.model_step(input, ctr.to(torch.float32))
        
        assert out_pred.shape == output[:,-3:].shape
        
        out_en = out_en.permute(0,2,3,1)  # bz, x, y, uvp

        loss1 = rel_error(out_pred[:].to(torch.float32), output[:,-3:]).mean()  # data loss
        loss2 = rel_error(out_de[:].to(torch.float32), input[:,-3:]).mean()  # encoder loss 1
        loss3 = rel_error(out_latent[:].to(torch.float32), output[:,-3:]).mean()  # encoder loss 2

        input_en = input_en.permute(0,2,3,1)  # bz, x, y, uvp
        trans_out = trans_out.permute(0,2,3,1)
        full_in = full_in[:,-3:].permute(0,2,3,1)
        full_out = full_out[:,-3:].permute(0,2,3,1)

        out_en_norm = out_en
        trans_out_norm = trans_out

        std = torch.tensor(std)
        mean = torch.tensor(mean)

        for channel in range(input_en.shape[-1]):
            input_en[:,:,:,channel] = input_en[:,:,:,channel]*std[channel] + mean[channel]
            out_en[:,:,:,channel] = out_en[:,:,:,channel]*std[channel] + mean[channel]
            trans_out[:,:,:,channel] = trans_out[:,:,:,channel]*std[channel] + mean[channel]
            
            full_in[:,:,:,channel] = full_in[:,:,:,channel]*std[channel] + mean[channel]
            full_out[:,:,:,channel] = full_out[:,:,:,channel]*std[channel] + mean[channel]

        error_pred = rel_error(trans_out.to(self.device), full_out.to(self.device)).mean()
        error_pred = rel_error(trans_out.to(self.device), full_out.to(self.device)).mean()
        error_output = rel_error(out_en.to(self.device), full_out.to(self.device)).mean()
        error_input = rel_error(input_en.to(self.device), full_in.to(self.device)).mean()
        
        t3 = default_timer()
        
        pde_loss_en = self.pde.physics_loss(input_en.to(self.device),out_en.to(self.device))

        t4 = default_timer()

        loss5 = pde_loss_en.to(torch.float32)
        loss6 = rel_error(trans_out_norm[:].to(self.device), out_en_norm[:].to(self.device)).mean()

        return loss1, loss2, loss3, loss5, loss6, trans_out, input_en, error_pred, error_input, error_output

    def model_step(self, input, ctr):

        t5 = default_timer()
        pred, x_de, trans_out, x_en = self.phys_inform_model(input, ctr)
        t6 = default_timer()
        
        assert pred.shape[1] == 3

        out_pred = pred.to(self.device)
        x_de = x_de.to(self.device)

        return out_pred, x_de, trans_out, x_en
    
    
    def phys_test(self, test_dataloader, ctr_test, logs):
        self.phys_inform_model.eval()
        args = logs['args']
        recall = args.recall_size

        test_log = PredLog(length=self.params.batch_size)
        
        with torch.no_grad():

            for x_test, y_test in test_dataloader:
                x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                ctr_test = ctr_test.to(self.device)
                    
                in_test = x_test[:,:3]
                out_test = y_test[:,:3]
                full_in_test = x_test[:,3:6]
                full_out_test = y_test[:,3:6]
                for i in range(recall-1):
                    in_test = torch.cat([in_test, x_test[:,i*6+6:i*6+9]], dim=1)
                    out_test = torch.cat([out_test, y_test[:,i*6+6:i*6+9]], dim=1)
                    full_in_test = torch.cat([full_in_test, x_test[:,i*6+9:i*6+12]], dim=1)
                    full_out_test = torch.cat([full_out_test, y_test[:,i*6+9:i*6+12]], dim=1)
                    # print(idx)
                mean = logs['mean']
                std = logs['std']

                loss1, loss2, loss3, loss5, loss6, trans_out, input_en, error_pred, error_input, error_output = self.pred_loss(in_test, ctr_test, out_test, full_in_test, full_out_test, mean, std)
                pde_loss_tr = self.pde.physics_loss(input_en.to(self.device),trans_out.to(self.device))
                loss4 = pde_loss_tr
                    
                test_log.update([loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.mean().item(), 
                                error_pred.item(), error_input.item(), error_output.item()])

            test_log.save_log(logs)

        print('--test | pred: {:1.2e}  input_en: {:1.2e}  output_en: {:1.2e} physics: {:1.2e} phys_encode: {:1.2e} hidden: {:1.2e} error_pred: {:1.2e} error_in: {:1.2e} error_out: {:1.2e}'
              .format(test_log.loss1.avg, test_log.loss2.avg, test_log.loss3.avg, test_log.loss4.avg, test_log.loss5.avg, test_log.loss6.avg,
                      test_log.loss7.avg, test_log.loss8.avg, test_log.loss9.avg))
        
    
    def train_step(self, loss1, loss2, loss3, loss4, loss5, loss6, logs):

        lambda1, lambda2, lambda3, lambda4, lambda5, lambda6 = self.params.lambda1, self.params.lambda2, self.params.lambda3, self.params.lambda4, self.params.lambda5, self.params.lambda6
        
        full = logs['full']
        
        t7 = default_timer()
        
        if full==1:
            loss_pred = lambda1 * loss1 + loss4 * lambda4
        elif full==0:
            loss_pred = lambda1 * loss1
            loss4 = lambda4 * loss4
            if loss4!=0:
                trans_grad_loss4 = torch.autograd.grad(loss4, self.phys_inform_model.trans.parameters(), only_inputs=True, retain_graph=True, allow_unused=False)

        loss5 = lambda5 * loss5

        if loss5!=0:
            encoder_grad_loss5 = torch.autograd.grad(loss5, self.phys_inform_model.model_en.parameters(), only_inputs=True, retain_graph=True, allow_unused=False)

        loss_pred.backward()  

        if full==0:
            if loss4!=0:
                for i, group in enumerate(self.phys_inform_model.trans.parameters()):
                    assert group.grad is not None
                    group.grad.data = group.grad.data + trans_grad_loss4[i]

        if loss5!=0:
            for j, group in enumerate(self.phys_inform_model.model_en.parameters()):
                assert group.grad is not None
                group.grad.data = group.grad.data + encoder_grad_loss5[j]

        self.phys_inform_optimizer.step()
        self.phys_inform_optimizer.zero_grad()
        
        t8 = default_timer()


    def load_state(self, pred_log):
        self.phys_inform_model.load_state_dict(pred_log)
        self.phys_inform_model.eval()
    
    def scheduler_step(self):
        # default setting
        self.phys_inform_scheduler.step()

    def set_init(self, full_in, full_out, state_nn, ctr_nn, out_nn):
        self.full_in = full_in.reshape(1,full_in.shape[0],full_in.shape[1],full_in.shape[2])
        self.full_out = full_out.reshape(full_out.shape[0],full_out.shape[1],full_out.shape[2],full_out.shape[3])
        self.in_nn = state_nn.reshape(1,state_nn.shape[0],state_nn.shape[1],state_nn.shape[2])
        self.ctr_nn = ctr_nn.reshape(1,ctr_nn.shape[0],ctr_nn.shape[1],ctr_nn.shape[2])
        self.out_nn = out_nn.reshape(out_nn.shape[0],out_nn.shape[1],out_nn.shape[2],out_nn.shape[3])
        
    def data_retrain_step(self, loss1, logs):
        lambda1 = self.params.lambda1
        loss_pred = loss1*lambda1

        encoder_grad_loss1 = torch.autograd.grad(loss_pred, self.phys_inform_model.model_en.parameters(), only_inputs=True, retain_graph=True, allow_unused=False)
        
        for i, group in enumerate(self.phys_inform_model.model_en.parameters()):
            # assert group.grad is not None
            if group.grad is None:
                    group.grad = torch.zeros_like(group)
            group.grad.data = group.grad.data + encoder_grad_loss1[i]
        
        self.phys_inform_optimizer.step()
        self.phys_inform_optimizer.zero_grad()

    def phys_retrain_step(self, loss4, logs):
        lambda4 = self.params.lambda4
        loss_pred = loss4*lambda4
        full = logs['full']

        if full == True:
            loss_pred.backward()
        elif full == False:
            trans_grad_loss4 = torch.autograd.grad(loss_pred, self.phys_inform_model.trans.parameters(), only_inputs=True, retain_graph=True, allow_unused=False)

            for i, group in enumerate(self.phys_inform_model.trans.parameters()):

                if group.grad is None:
                    group.grad = torch.zeros_like(group)

                group.grad.data = group.grad.data + trans_grad_loss4[i]
        
        self.phys_inform_optimizer.step()
        self.phys_inform_optimizer.zero_grad()
    
    def retrain_loss(self, ipt, ctr, opt, full_in, full_out, mean, std):

        input = self.phys_inform_model.model_de.output(ipt).to(self.device).to(torch.float32)  # form the input data with downsample      bz, uvp*n, ox, oy
        output = self.phys_inform_model.model_de.output(opt).to(self.device).to(torch.float32)  # form the output data with downsample

        out_pred, out_de, trans_out, input_en = self.model_step(input, ctr.to(torch.float32))

        assert out_pred.shape == output[:,-3:].shape
        
        input_en = input_en.permute(0,2,3,1)  # bz, x, y, uvp
        trans_out = trans_out.permute(0,2,3,1)
        full_in = full_in[:,-3:].permute(0,2,3,1)
        full_out = full_out[:,-3:].permute(0,2,3,1)

        std = torch.tensor(std)
        mean = torch.tensor(mean)

        for channel in range(input_en.shape[-1]):
            input_en[:,:,:,channel] = input_en[:,:,:,channel]*std[channel] + mean[channel]
            trans_out[:,:,:,channel] = trans_out[:,:,:,channel]*std[channel] + mean[channel]
            
            full_in[:,:,:,channel] = full_in[:,:,:,channel]*std[channel] + mean[channel]
            full_out[:,:,:,channel] = full_out[:,:,:,channel]*std[channel] + mean[channel]

        return trans_out, input_en, out_pred, output

    def phys_retrain(self, epoch, train_dataloader, ctr_train, logs):
        
        self.phys_inform_model.train()
        args = logs['args']
        recall = args.recall_size
        t1 = default_timer()
        retrain_log = PredLog(length=self.params.batch_size)

        for x_train, y_train in train_dataloader:

            self.phys_inform_model.zero_grad()

            x_train, y_train = x_train.to(self.device), y_train.to(self.device)  # bz, uvp*n*2, x, y
            ctr_train = ctr_train.to(self.device)  # bz, x, y, 1
                
            in_train = x_train[:,:3]
            out_train = y_train[:,:3]
            full_in = x_train[:,3:6]
            full_out = y_train[:,3:6]
            for i in range(recall-1):
                in_train = torch.cat([in_train, x_train[:,i*6+6:i*6+9]], dim=1)   # bz, uvp*n, x, y
                out_train = torch.cat([out_train, y_train[:,i*6+6:i*6+9]], dim=1)
                full_in = torch.cat([full_in, x_train[:,i*6+9:i*6+12]], dim=1)
                full_out = torch.cat([full_out, y_train[:,i*6+9:i*6+12]], dim=1)
            
            mean = logs['mean']
            std = logs['std']
            
            trans_out, input_en, out_pred, output = self.retrain_loss(in_train, ctr_train, out_train, full_in, full_out, mean, std)

            pde_loss_tr = self.pde.physics_loss(input_en.to(self.device),trans_out.to(self.device))
            loss4 = pde_loss_tr.to(torch.float32)
                
            self.phys_retrain_step(loss4.to(self.device), logs)

            retrain_log.update([loss4.item()])

        retrain_log.save_phys_retrain(logs)

        self.scheduler_step()
        t2 = default_timer()

        print('# {} retrain: {:1.2f} | physics: {:1.2e}'
              .format(epoch, t2-t1, retrain_log.loss1.avg))
        
        
    def data_retrain(self, epoch, train_dataloader, ctr_train, logs):
        
        self.phys_inform_model.train()
        args = logs['args']
        recall = args.recall_size
        t1 = default_timer()
        retrain_log = PredLog(length=self.params.batch_size)

        for x_train, y_train in train_dataloader:

            self.phys_inform_model.zero_grad()

            x_train, y_train = x_train.to(self.device), y_train.to(self.device)  # bz, uvp*n*2, x, y
            ctr_train = ctr_train.to(self.device)  # bz, x, y, 1
                
            in_train = x_train[:,:3]
            out_train = y_train[:,:3]
            full_in = x_train[:,3:6]
            full_out = y_train[:,3:6]
            for i in range(recall-1):
                in_train = torch.cat([in_train, x_train[:,i*6+6:i*6+9]], dim=1)   # bz, uvp*n, x, y
                out_train = torch.cat([out_train, y_train[:,i*6+6:i*6+9]], dim=1)
                full_in = torch.cat([full_in, x_train[:,i*6+9:i*6+12]], dim=1)
                full_out = torch.cat([full_out, y_train[:,i*6+9:i*6+12]], dim=1)
            
            mean = logs['mean']
            std = logs['std']
            
            trans_out, input_en, out_pred, output = self.retrain_loss(in_train, ctr_train, out_train, full_in, full_out, mean, std)
                        
            loss1 = rel_error(out_pred[:].to(torch.float32), output[:,-3:]).mean()
            self.data_retrain_step(loss1.to(self.device), logs)
            retrain_log.update([loss1.item()])

        retrain_log.save_data_retrain(logs)

        t2 = default_timer()

        print('# {} retrain_data: {:1.2f} | pred: {:1.2e}'
              .format(epoch, t2-t1, retrain_log.loss1.avg))

class NSEModel_FNO(NSEModel):
    def __init__(self, shape, dt, args):
        super().__init__(shape, dt, args)
        self.Re = args.re
        self.Lx = args.lx
        self.Ly = args.ly
        # self.Lx = 2.2
        # self.Ly = 0.41
        self.set_model()
    
    def cal_1step(self, data, arg):
        obs, Cd, Cl, ctr = data.get_data()

        N0, nt = obs.shape[0], obs.shape[1] - 1  # nt=25, N0=20
        
        nx, ny = self.shape
        print(ctr.shape)
        ctr = ctr.reshape(N0, nt, 1, 1, 3).repeat([1, 1, nx, ny, 1]).reshape(-1, nt, nx, ny, 3)

        Cd = Cd.reshape(N0, nt, 1, 1, 1).repeat([1, 1, nx, ny, 1]).reshape(-1, nt, nx, ny, 1)
        Cl = Cl.reshape(N0, nt, 1, 1, 1).repeat([1, 1, nx, ny, 1]).reshape(-1, nt, nx, ny, 1)
        Cd_mean, Cd_std = Cd.mean(), Cd.std()
        Cl_mean, Cl_std = Cl.mean(), Cl.std()
        obs0_mean, obs0_std = obs[:,:,:,:,0].mean(), obs[:,:,:,:,0].std()
        obs1_mean, obs1_std = obs[:,:,:,:,1].mean(), obs[:,:,:,:,1].std()
        obs2_mean, obs2_std = obs[:,:,:,:,2].mean(), obs[:,:,:,:,2].std()
        ctr_mean, ctr_std = ctr.mean(), ctr.std()

        Cd = (Cd - Cd_mean) / Cd_std
        Cl = (Cl - Cl_mean) / Cl_std
        obs[:,:,:,:,0] = (obs[:,:,:,:,0] - obs0_mean) / obs0_std
        obs[:,:,:,:,1] = (obs[:,:,:,:,1] - obs1_mean) / obs1_std
        obs[:,:,:,:,2] = (obs[:,:,:,:,2] - obs2_mean) / obs2_std
        ctr = (ctr-ctr_mean) / ctr_std
        
        out_nn, Lpde_obs, Lpde_pred = torch.zeros(N0, nt, nx, ny, 3), torch.zeros(N0, nt, nx, ny, 2), torch.zeros(N0, nt, nx, ny, 2)
        Cd_nn, Cl_nn = torch.zeros(N0, nt), torch.zeros(N0, nt)
        error_1step, error_Cd, error_Cl = torch.zeros(N0, nt), torch.zeros(N0, nt), torch.zeros(N0, nt)

        logs = dict()

        logs['error_phy'] = []
        logs['error_data'] = []

        with torch.no_grad():
            for k in range(nt):
                t1 = default_timer()
                out_nn[:, k], Cd_nn[:, k], Cl_nn[:, k], mod_pred, _, _, _ = self.model_step(obs[:, k], ctr[:, k])
                Lpde_pred[:, k] = ((Lpde(obs[:, k], out_nn[:, k], self.dt, Lx = self.Lx, Ly = self.Ly, Re = self.Re, arg = arg) + mod_pred) ** 2)

                mod_obs = self.phys_model(obs[:, k], ctr[:, k], obs[:, k+1])
                Lpde_obs[:, k] = ((Lpde(obs[:, k], obs[:, k+1], self.dt, Lx = self.Lx, Ly = self.Ly, Re = self.Re, arg = arg) + mod_obs) ** 2)
                
                error_1step[:, k] = rel_error(out_nn[:, k], obs[:, k+1]) 
                error_Cd[:, k] = ((Cd_nn[:, k][0] - Cd[:, k, 0, 0][0])**2)
                error_Cl[:, k] = ((Cl_nn[:, k][0] - Cl[:, k, 0, 0][0])**2)
                error_sum  = error_1step + error_Cd + error_Cl
                t2 = default_timer()
                if k % 1 == 0:
                    print(f'# {k} | {t2 - t1:1.2f}: error_Cd: {error_Cd[:, k].mean():1.4f} | error_Cl: {error_Cl[:, k].mean():1.4f} | error_state: {error_1step[:, k].max():1.4f}\
                        | pred_Lpde: {Lpde_pred[:, k].mean():1.4f} | obs_Lpde: {Lpde_obs[:, k].mean():1.4f}')

        logs['error_phy'].append(error_sum)
        torch.save(logs, '/haodong/plot/phy_2')

        return error_1step, Lpde_obs, Lpde_pred

    
    def process_4(self, data, ctr, labels, logs):
        recall = logs['args'].recall_size
        in_test = data[:,:3]
        out_test = labels[:,:3]
        full_in_test = data[:,3:6]
        full_out_test = labels[:,3:6]
        for i in range(recall-1):
            in_test = torch.cat([in_test, data[:,i*6+6:i*6+9]], dim=1)
            out_test = torch.cat([out_test, labels[:,i*6+6:i*6+9]], dim=1)
            full_in_test = torch.cat([full_in_test, data[:,i*6+9:i*6+12]], dim=1)
            full_out_test = torch.cat([full_out_test, labels[:,i*6+9:i*6+12]], dim=1)

        mean = logs['mean']
        std = logs['std']
        
        N0, nt = 1, data.shape[0]
        nx, ny = self.shape
        print(f'N0: {N0}, nt: {nt}, nx: {nx}, ny: {ny}')
        out, trans_out, Lpde_pred = torch.zeros(nt, 3, 7, 7), torch.zeros(nt, 3, 32, 32), torch.zeros(nt)
        error_pred, error_full = torch.zeros(nt), torch.zeros(nt)
        step = 10
        logs = dict()
        logs['error_phy_12'] = []

        with torch.no_grad():
            self.set_init(full_in_test[0], full_out_test, in_test[0], ctr, out_test)
            self.in_nn =  self.phys_inform_model.model_de.output(self.in_nn).to(torch.float32)  # form the input data with downsample
            self.out_nn =  self.phys_inform_model.model_de.output(self.out_nn).to(torch.float32)  # form the input data with downsample
            data_loss = []
                
            for k in range(step):
                t1 = default_timer()

                out[k, :], x_de, trans_out[k,:], x_en = self.model_step(self.in_nn, self.ctr_nn.to(torch.float32))

                error_pred[k] = rel_error(out[k, :].to(torch.float32), self.out_nn[:,-3:][k, :]).mean()
                error_full[k] = rel_error(trans_out[k, :].to(torch.float32), self.full_out[:,-3:][k, :]).mean()

                data_loss.append(error_pred[k])

                input_en = torch.zeros_like(x_en[0]).permute(1,2,0)
                out_en = torch.zeros_like(trans_out[k,:]).permute(1,2,0)
                full_in = self.full_in[:,-3:][0].permute(1,2,0)
                full_out = self.full_out[:,-3:][k,:].permute(1,2,0)

                for channel in range(input_en.shape[-1]):
                    input_en[:,:,channel] = x_en[0].permute(1,2,0)[:,:,channel]*std[channel].to('cpu') + mean[channel].to('cpu')
                    out_en[:,:,channel] = trans_out[k,:].permute(1,2,0)[:,:,channel]*std[channel].to('cpu') + mean[channel].to('cpu')
                    
                    full_in[:,:,channel] = self.full_in[:,-3:][0].permute(1,2,0)[:,:,channel]*std[channel].to('cpu') + mean[channel].to('cpu')
                    full_out[:,:,channel] = self.full_out[:,-3:][k,:].permute(1,2,0)[:,:,channel]*std[channel].to('cpu') + mean[channel].to('cpu')

                input_en = torch.unsqueeze(input_en, dim=0)
                out_en = torch.unsqueeze(out_en, dim=0)
                Lpde_pred[k] = self.pde.physics_loss(input_en.to(self.device),out_en.to(self.device))

                self.in_nn[0] = torch.cat((self.in_nn[0,3:],out[k, :]),dim=0)

        return out, self.out_nn, data_loss

