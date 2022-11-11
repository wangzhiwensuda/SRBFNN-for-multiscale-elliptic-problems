import torch
import torch.optim as optims
from torch import nn
import torch.utils.data as Data
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import time
import torch.optim.lr_scheduler as lr_scheduler
torch.manual_seed(1)
torch.set_printoptions(precision=8)
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')



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

def get_bd_loss(net):
    xa = torch.zeros(1).to(device)
    xb = torch.ones(1).to(device)
    la = (net(xa)-1)**2
    lb = (net(xb)-1)**2
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
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True,)[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)


def get_rel_err(net,eps):
    net = net.cpu()
    x = torch.linspace(0, 1, 10001).view(-1,1)
    x.requires_grad_(True)
    u = net(x)
    ux = gradients(u,x,order=1)
    u,ux = u.detach().numpy().flatten(),ux.detach().numpy().flatten()
    u_FDM,ux_FDM = exact_solution(eps)
    L2_err = np.sqrt(((u - u_FDM) ** 2).sum()) / np.sqrt((u_FDM ** 2).sum())
    H1_err = np.sqrt((((ux - ux_FDM) ** 2).sum() + ((u - u_FDM) ** 2).sum())) / (
        np.sqrt((ux_FDM ** 2).sum() + (u_FDM ** 2).sum()))
    L_inf_err = np.max(abs(u - u_FDM)) / np.max(abs(u_FDM))
    return L2_err, H1_err, L_inf_err



class DGM(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, m=16, actv=nn.Tanh()):
        super(DGM, self).__init__()
        self.actv = actv
        self.linear_input = nn.Linear(input_dim, m)
        self.linear2 = nn.Linear(m, m)
        self.linear3 = nn.Linear(m, m)
        self.linear4 = nn.Linear(m, m)
        self.linear5 = nn.Linear(m, m)
        self.linear_output = nn.Linear(m, output_dim)

    def forward(self, x):
        y = self.actv(self.linear_input(x))
        y = y + self.actv(self.linear3(self.actv(self.linear2(y))))
        y = y + self.actv(self.linear5(self.actv(self.linear4(y))))
        output = self.linear_output(y)

        return output
	




# 定义训练函数
def train(net, epochs, data_iter, lr, eps):
    print('Training on %s' % device)
    optimizer = optims.Adam(net.parameters(), lr)
    t_all = 0.0
    net = net.to(device=device)
    for epoch in range(1, epochs + 1):
        l_r_sum = 0.0
        l_bd_sum = 0.0
        t1 = time.time()
        for x,a,f in data_iter:
            x = x.to(device=device)
            a = a.to(device=device)
            f = f.to(device)
            u = net(x)
            ux = gradients(u,x,order=1)
            l_r = ((gradients(a*ux,x,order=1)+f)**2).mean()
            l_bd = get_bd_loss(net)
            l = l_r + 20 * l_bd
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
        if epoch %20==0:
            #torch.save(net.state_dict(), 'model/1d_exam2_%.3f.pth' % (eps))
            net = net.cpu()
            L2, H1, L_inf = get_rel_err(net,eps)
            print('L2:', L2, 'H1:', H1, 'L_inf:', L_inf)
            print('--------lr:{0:.6f}'.format(optimizer.param_groups[0]['lr']))
            net = net.to(device)

def show(eps):
    x =torch.linspace(0,1,10001).view(-1,1)
    x.requires_grad_(True)
    u = net(x).flatten()
    ux = gradients(u,x,order=1)
    x,u,ux = x.detach().numpy(),u.detach().numpy(),ux.detach().numpy()
    u_FDM,ux_FDM = exact_solution(eps)
    plt.plot(x,u,'r')
    plt.plot(x,u_FDM)
    plt.show()

if __name__ == "__main__":
    batchsize = 2048
    epochs = 3000
    lr = 0.0001
    eps = 0.5
    dataloader = Dataloader(n=10000, eps=eps,batchsize=batchsize)
    net = DGM(input_dim=1,output_dim=1,m=16) #[16,16,32,32,64,64]
    if os.path.exists('model/1d_exam2_%.3f.pth' % (eps)):
        print('---加载已训练网络----')
        #net.load_state_dict(torch.load('model/1d_exam2_%.3f.pth' % (eps)))
    train(net, epochs, dataloader, lr, eps)
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
