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
    ax = (2 + torch.sin(2 * math.pi * x / eps) * torch.cos(2 * math.pi * x))
    f = torch.ones_like(x)
    return x, ax, f

def Dataloader(n=10000,eps=0.5,batchsize=2000):
    x,ax,f = SampleData(n,eps)
    dataset = Data.TensorDataset(x, ax,f)
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




# Neural Network
class PINN(torch.nn.Module):
    def __init__(self,input_dim=1,hidden_layers=32):
        super(PINN, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_layers),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_layers, hidden_layers),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_layers, hidden_layers),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_layers, hidden_layers),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_layers, 1)
        )

    def forward(self, x):
        return self.net(x)






# 定义训练函数
def train(net, epochs, data_iter, lr, eps):
    print('Training on %s' % device)
    optimizer = optims.Adam(net.parameters(), lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    t_all = 0.0
    net = net.to(device=device)
    for epoch in range(1, epochs + 1):
        l_r_sum = 0.0
        l_bd_sum = 0.0
        t1 = time.time()
        for x, ax, f in data_iter:
            x = x.to(device=device)
            ax = ax.to(device=device)
            f = f.to(device)
            u = net(x)
            ux = gradients(u,x,order=1)
            l_r = ((gradients(ax*ux,x,order=1)+f)**2).mean()
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
        if (optimizer.param_groups[0]['lr'] > 0.0001):
            scheduler.step()
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
    lr = 0.001
    eps = 0.002
    dataloader = Dataloader(n=10000, eps=eps,batchsize=batchsize)
    net = PINN(input_dim=1,hidden_layers=64) #[16,16,32,32,64,64]
    if os.path.exists('model/1d_exam2_%.3f.pth' % (eps)):
        print('---加载已训练网络,eps:%.3f----'%eps)
        #net.load_state_dict(torch.load('model/1d_exam2_%.3f.pth' % (eps)))
    train(net, epochs, dataloader, lr, eps)
    net = net.cpu()
    #show(eps)
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %d" % (total))
    L2, H1, L_inf = get_rel_err(net,eps)
    print('L2:', L2, 'H1:', H1, 'L_inf:', L_inf)


"""    m   params        L2              H1            L_inf
0.5:  16    865        L2: 0.002069622547413297 H1: 0.021215529196052556 L_inf: 0.004267370938090232
0.1:   16    865       L2: 0.006008687935775626 H1: 0.04868987886457674 L_inf: 0.007903846469107566
0.05:  32    3265       L2: 0.005914791902379552 H1: 0.04922858649232228 L_inf: 0.007645556025334873
0.01:  32    3265       L2: 0.005882915593524392 H1: 0.04937423728524035 L_inf: 0.007520637908192104
0.005:  64   12673       L2: 0.005835008003244552 H1: 0.04925444037727992 L_inf: 0.007585345603634334
0.002:  64   12673       L2: 0.00591223098864397 H1: 0.049157520494394744 L_inf: 0.007693095823247554
"""
"""
params:  241,241,865,865,3265,3265

"""