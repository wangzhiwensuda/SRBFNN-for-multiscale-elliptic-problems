import torch
import torch.optim as optims
from torch import nn
import torch.utils.data as Data
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as tnf
import math
import time
import torch.optim.lr_scheduler as lr_scheduler
torch.manual_seed(1)
torch.set_printoptions(precision=8)
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')



# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 加载数据
def SampleData(n=10000,eps=0.5):
    x = torch.linspace(0,1,n).view(-1,1)
    x.requires_grad_(True)
    a = (2 + torch.sin(2 * math.pi * x / eps) * torch.cos(2 * math.pi * x))
    f = torch.ones_like(x)
    return x,a,f

    
def Dataloader(n=10000,eps=0.5,batchsize=2000):
    x,a,f = SampleData(n,eps)
    dataset = Data.TensorDataset(x,a,f)
    Dataloader = Data.DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True)
    return Dataloader

def get_bd_loss(net,freq):
    xa = torch.zeros(1).to(device)
    xb = torch.ones(1).to(device)
    la = (net(xa,scale=freq)-1)**2
    lb = (net(xb,scale=freq)-1)**2
    return la+ lb

def exact_solution(eps):
    u_FDM = np.load('1d_FDM/1d_q2_%.3f.npy' % (eps)).flatten()
    ux_FDM = np.zeros_like(u_FDM)
    ux_FDM[1:-1] = (u_FDM[2:] - u_FDM[:-2]) /0.0002
    ux_FDM[0] = -1.5 * u_FDM[0] /0.0002 + 2 * u_FDM[1] / 0.0002 - 0.5 * u_FDM[2] / 0.0002
    ux_FDM[-1] = 1.5 * u_FDM[-1] / 0.0002 - 2 * u_FDM[-2] / 0.0002 + 0.5 * u_FDM[-3] / 0.0002
    return u_FDM, ux_FDM

def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True,)[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)


def get_rel_err(net,eps):
    net = net.to(device)
    x = torch.linspace(0, 1, 10001).view(-1,1).to(device)
    x.requires_grad_(True)
    u = net(x,scale=freq)
    ux = gradients(u,x,order=1)
    u,ux = u.cpu().detach().numpy().flatten(),ux.cpu().detach().numpy().flatten()
    u_FDM,ux_FDM = exact_solution(eps)
    L2_err = np.sqrt(((u - u_FDM) ** 2).sum()) / np.sqrt((u_FDM ** 2).sum())
    H1_err = np.sqrt((((ux - ux_FDM) ** 2).sum() + ((u - u_FDM) ** 2).sum())) / (
        np.sqrt((ux_FDM ** 2).sum() + (u_FDM ** 2).sum()))
    L_inf_err = np.max(abs(u - u_FDM)) / np.max(abs(u_FDM))
    return L2_err, H1_err, L_inf_err

class my_actFunc(nn.Module):
    def __init__(self, actName='linear'):
        super(my_actFunc, self).__init__()
        self.actName = actName
    def forward(self, x_input):
        if str.lower(self.actName) == 'tanh':
            out_x = torch.tanh(x_input)
        elif str.lower(self.actName) == 'srelu':
            out_x = tnf.relu(x_input)*tnf.relu(1-x_input)
        elif str.lower(self.actName) == 's2relu':
            out_x = tnf.relu(x_input)*tnf.relu(1-x_input)*torch.sin(2*np.pi*x_input)
        else:
            out_x = x_input
        return out_x

class MscaleDNN(nn.Module):
    def __init__(self, indim=1, outdim=1, hidden_units=None, actName2in='s2relu', actName='s2relu',
                 actName2out='linear'):
                 
        super(MscaleDNN, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.dense_layers = nn.ModuleList()

        input_layer = nn.Linear(in_features=indim, out_features=hidden_units[0])
        nn.init.xavier_normal_(input_layer.weight)
        nn.init.uniform_(input_layer.bias, -1, 1)
        self.dense_layers.append(input_layer)

        for i_layer in range(len(hidden_units)-1):
            hidden_layer = nn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer+1])
            nn.init.xavier_normal_(hidden_layer.weight)
            nn.init.uniform_(hidden_layer.bias, -1, 1)
            self.dense_layers.append(hidden_layer)

        out_layer = nn.Linear(in_features=hidden_units[-1], out_features=outdim)
        nn.init.xavier_normal_(out_layer.weight)
        nn.init.uniform_(out_layer.bias, -1, 1)
        self.dense_layers.append(out_layer)

    def forward(self, inputs, scale=None):  #scale=np.array([1,1,2,...,99])
        # ------ dealing with the input data ---------------
        dense_in = self.dense_layers[0]
        H = dense_in(inputs)  #num_points*300,,,点的个数*输入层神经元个数
        Unit_num = int(self.hidden_units[0] / len(scale))  #300,100
        mixcoe = np.repeat(scale, Unit_num) #scale=np.array([1,1,1,1,1,1,2,2,2,3,3,3,4,4,4,5...,99])


        mixcoe = mixcoe.astype(np.float32)
        torch_mixcoe = torch.from_numpy(mixcoe)
        if inputs.is_cuda:
            torch_mixcoe = torch_mixcoe.to(device)
        #print('-----',H,torch_mixcoe)
        H = self.actFunc_in(H*torch_mixcoe)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units)-1):
            H_pre = H
            dense_layer = self.dense_layers[i_layer+1]
            H = dense_layer(H)
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        dense_out = self.dense_layers[-1]
        H = dense_out(H)
        H = self.actFunc_out(H)
        return H
	






# 定义训练函数
def train(net, epochs, data_iter, lr, eps,freq):
    print('Training on %s' % device)
    optimizer = optims.Adam(net.parameters(), lr)
    t_all = 0.0
    net = net.to(device=device)
    use_L2_loss = False
    for epoch in range(1, epochs + 1):
        l_r_sum = 0.0
        l_bd_sum = 0.0
        t1 = time.time()
        for x,a,f in data_iter:
            x = x.to(device=device)
            a = a.to(device=device)
            f = f.to(device)
            u = net(x,scale=freq)
            ux = gradients(u,x,order=1)
            if use_L2_loss:
                l_r = ((gradients(a*ux,x,order=1)+f)**2).mean()
                lam_b = 20.0
            else:
                l_r = torch.mean(0.5*ux*a*ux-f*u)
                lam_b = 800.0
            l_bd = get_bd_loss(net,freq)
            l = l_r + lam_b * l_bd
            optimizer.zero_grad()
            l.backward(retain_graph=True)
            optimizer.step()
            l_r_sum += l_r.cpu().item()
            l_bd_sum += l_bd.cpu().item()
        t2 = time.time()
        t_all += t2 - t1
        l_sums = l_r_sum  + l_bd_sum
        print('eps=%.3f,t_all=%.3f,t_epoch=%.3f,epoch %d,l_r:%f,l_bd:%f,l_all:%f'
              % (eps, t_all, t2 - t1, epoch, l_r_sum, l_bd_sum, l_sums))
        if (epoch%100==0)&(optimizer.param_groups[0]['lr']>1e-5):
            optimizer.param_groups[0]['lr']=0.1*optimizer.param_groups[0]['lr']      
        if epoch %20==0:
            torch.save(net.state_dict(), 'model/1d_exam2_%.3f.pth' % (eps))
            net = net.cpu()
            L2, H1, L_inf = get_rel_err(net,eps)
            print('L2:', L2, 'H1:', H1, 'L_inf:', L_inf)
            print('--------lr:{0:.6f}'.format(optimizer.param_groups[0]['lr']))
            net = net.to(device)

def show(eps):
    x =torch.linspace(0,1,10001).view(-1,1)
    x.requires_grad_(True)
    u = net(x,freq).flatten()
    ux = gradients(u,x,order=1)
    x,u,ux = x.detach().numpy(),u.detach().numpy(),ux.detach().numpy()
    u_FDM,ux_FDM = exact_solution(eps)
    plt.plot(x,u,'r')
    plt.plot(x,u_FDM)
    plt.legend(['nn','FDM'])
    plt.show()

if __name__ == "__main__":
    batchsize = 2048
    epochs = 3000
    lr = 0.0001
    eps = 0.05
    freq = np.arange(int(1/eps))
    freq[0] += 1.0 
    dataloader = Dataloader(n=10000, eps=eps,batchsize=batchsize)
    net = MscaleDNN(indim=1, outdim=1, hidden_units=(1000, 200, 150, 150,100,50, 50), actName2in='s2relu', actName='s2relu',
                 actName2out='linear') 
    if os.path.exists('model/1d_exam2_%.3f.pth' % (eps)):
        print('---加载已训练网络----')
        net.load_state_dict(torch.load('model/1d_exam2_%.3f.pth' % (eps)))
    train(net, epochs, dataloader, lr, eps,freq)
    net = net.cpu()
    show(eps)
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %d" % (total))
    L2, H1, L_inf = get_rel_err(net,eps)
    print('L2:', L2, 'H1:', H1, 'L_inf:', L_inf)

"""    m   params        L2              H1            L_inf
0.5:  16    1137       L2: 0.002004104704343646 H1: 0.0213308312754096 L_inf: 0.004205756067252141
0.1:   16    1137      L2: 0.0060563957650701 H1: 0.04877994557854111 L_inf: 0.007895107228979273
0.05:  32    4321       L2: 0.005981856744959118 H1: 0.04926578096487784 L_inf: 0.007665538679153879
0.01:  32    4321      L2: 0.005932781721565501 H1: 0.04939964901589382 L_inf: 0.00754555052138904
0.005:  64   16833     L2: 0.005848410048824204 H1: 0.04928577491636954 L_inf: 0.007632492853264291
0.002:  64   16833     L2: 0.005896126610497069 H1: 0.04914163504790333 L_inf: 0.007697881205147374
"""
