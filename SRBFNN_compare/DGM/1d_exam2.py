import torch
import torch.optim as optims
from torch import nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import time
import argparse

torch.manual_seed(1)
torch.set_printoptions(precision=8)




# generate training data
def dataloader(num_points, batch_size, eps):
    x = torch.rand(num_points,1)
    x.requires_grad_(True)
    a = 2 + torch.sin(2 * math.pi * x / eps)*torch.cos(2*math.pi*x)
    f = torch.ones_like(x)
    dataset = Data.TensorDataset(x, a,f)
    data_iter = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_iter


#load the FDM solution
def generate_FDM(eps):
    if os.path.exists('1d_FDM/1d_exam2_%.3f.npy'% (eps)): 
        u_FDM = np.load('1d_FDM/1d_exam2_%.3f.npy' % (eps)).flatten()
        ux_FDM = np.zeros_like(u_FDM)
        ux_FDM[1:-1] = (u_FDM[2:] - u_FDM[:-2]) / 0.0002
        ux_FDM[0] = -1.5 * u_FDM[0] / 0.0001 + 2 * u_FDM[1] / 0.0001 - 0.5 * u_FDM[2] / 0.0001
        ux_FDM[-1] = 1.5 * u_FDM[-1] / 0.0001 - 2 * u_FDM[-2] / 0.0001 + 0.5 * u_FDM[-3] / 0.0001
    else:
        raise ValueError("no the FDM solution")
    return u_FDM, ux_FDM


#calculate the boundary loss
def get_bd_loss(net,device):
    xa = torch.zeros(1).to(device)
    xb = torch.ones(1).to(device)
    la = (net(xa)-1)**2
    lb = (net(xb)-1)**2
    return la+ lb

#use autograd to calculate the gradients
def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True,only_inputs=True)[0]                          
    else:
        return gradients(gradients(u, x), x, order=order - 1)


#calculate the three relative errors
def get_err(net,eps):
    net = net.cpu()
    x = torch.linspace(0, 1, 10001).view(-1,1)
    x.requires_grad_(True)
    u = net(x)
    ux = gradients(u,x,order=1)
    u,ux = u.detach().numpy().flatten(),ux.detach().numpy().flatten()
    u_FDM,ux_FDM = generate_FDM(eps)
    L2_err = np.sqrt(((u - u_FDM) ** 2).sum()) / np.sqrt((u_FDM ** 2).sum())
    H1_err = np.sqrt((((ux - ux_FDM) ** 2).sum() + ((u - u_FDM) ** 2).sum())) / (
        np.sqrt((ux_FDM ** 2).sum() + (u_FDM ** 2).sum()))
    L_inf_err = np.max(abs(u - u_FDM)) / np.max(abs(u_FDM))
    return L2_err, H1_err, L_inf_err


#The model of DGM
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
	




#the training process of DGM
def train(net, epochs, data_iter,device, lr, eps,lam_b):
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
            l_bd = get_bd_loss(net,device)
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
        if epoch %20==0:
            #torch.save(net.state_dict(), 'model/1d_exam2_%.3f.pth' % (eps))
            L2, H1, L_inf = get_err(net,eps)
            print('Rel. L2:', L2,'Rel. L_inf:', L_inf, 'Rel. H1:', H1 )
            print('--------lr:{0:.6f}'.format(optimizer.param_groups[0]['lr']))
            net = net.to(device)
            
#show the solution 
def show(net,eps):
    net = net.cpu()
    x =torch.linspace(0,1,10001).view(-1,1)
    x.requires_grad_(True)
    u = net(x).flatten()
    ux = gradients(u,x,order=1)
    x,u,ux = x.detach().numpy(),u.detach().numpy(),ux.detach().numpy()
    u_FDM,ux_FDM = generate_FDM(eps)
    plt.plot(x,u)
    plt.plot(x,u_FDM)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend(['DGM','FDM'])
    plt.show()
    plt.plot(x,ux,'r')
    plt.plot(x,ux_FDM)
    plt.xlabel('x')
    plt.ylabel(r'$u_x$')
    plt.legend(['DGM','FDM'])
    plt.show()


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    eps = args.eps
    num_points = args.num_points
    data_iter = dataloader(num_points,batch_size,eps)
    m = args.m
    lam_b = args.lam_b
    net = DGM(input_dim=1,output_dim=1,m=m) #[16,16,32,32,64,64]
    if args.pretrained:
        if os.path.exists('model/1d_exam2_%.3f.pth' % (eps)):
            net.load_state_dict(torch.load('model/1d_exam2_%.3f.pth' % (eps)))
            print('load the trained model')
        else:
            print('no trained model')
    train(net, epochs, data_iter,device, lr, eps,lam_b)
    net = net.cpu()
    show(net,eps)
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameters for DGM: %d" % (total))
    L2, H1, L_inf = get_err(net,eps)
    print('Rel. L2:', L2,'Rel. L_inf:', L_inf, 'Rel. H1:', H1 )


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--eps', type=float, default=0.5) #[0.5,0.1,0.05,0.01,0.005,0.002]
    parser.add_argument('--m', type=int, default=16) #[16,16,32,32,64,64], the number of hidden layers
    parser.add_argument('--batch_size',type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_points', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lam_b', type=float, default=20.0)
    parser.add_argument('--pretrained', type=str, default=False)
    args = parser.parse_args()
    main(args)

"""    m   params       
0.5:  16    1137       L2: 0.002004104704343646 H1: 0.0213308312754096 L_inf: 0.004205756067252141
0.1:   16    1137      L2: 0.0060563957650701 H1: 0.04877994557854111 L_inf: 0.007895107228979273
0.05:  32    4321       L2: 0.005981856744959118 H1: 0.04926578096487784 L_inf: 0.007665538679153879
0.01:  32    4321      L2: 0.005932781721565501 H1: 0.04939964901589382 L_inf: 0.00754555052138904
0.005:  64   16833     L2: 0.005848410048824204 H1: 0.04928577491636954 L_inf: 0.007632492853264291
0.002:  64   16833     L2: 0.005896126610497069 H1: 0.04914163504790333 L_inf: 0.007697881205147374
"""
