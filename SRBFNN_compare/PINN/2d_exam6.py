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
import torch.optim.lr_scheduler as lr_scheduler
from mpl_toolkits.axes_grid1 import make_axes_locatable
torch.manual_seed(1)
torch.set_printoptions(precision=8)


# generate training data
def dataloader(h, batch_size, eps):
    x0 = torch.linspace(0, 1, int(1 / h + 1))
    x,y = torch.meshgrid(x0,x0)
    x,y = x.flatten(),y.flatten()
    x,y = x.view(-1,1),y.view(-1,1)
    
    dataset = Data.TensorDataset(x,y)
    data_iter = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_iter

#load the FDM solution
def generate_FDM(eps):
    if os.path.exists('2d_FDM/2d_exam6_FDM_%.3f.npy'% (eps)): 
        u_FDM = np.load('2d_FDM/2d_exam6_FDM_%.3f.npy' % (eps))[::2,::2]
        ux_FDM = np.load('2d_FDM/2d_exam6_FDM_x_%.3f.npy' % (eps))[::2,::2][1:-1,1:-1]
        uy_FDM = np.load('2d_FDM/2d_exam6_FDM_y_%.3f.npy' % (eps))[::2,::2][1:-1,1:-1]
    else:
        raise ValueError("no the FDM solution")
    return u_FDM, ux_FDM,uy_FDM

#calculate the boundary loss
def get_bd_loss(net,device,batch_size_bd=512):
    x = torch.linspace(0,1,batch_size_bd).view(-1,1).to(device)
    y = torch.zeros_like(x).to(device)
    X_up = torch.cat([x,y+1],dim=1) #(x,1)
    X_down = torch.cat([x,y],dim=1) #(x,0)
    X_left = torch.cat([y,x],dim=1) ##(0,y)
    X_right = torch.cat([y+1,x],dim=1) #(1,y)
    l_up = ((net(X_up)-0)**2).mean()
    l_down = ((net(X_down)-0)**2).mean()
    l_left = ((net(X_left)-0)**2).mean()
    l_right = ((net(X_right)-0)**2).mean()
    return l_up+l_down+l_left+l_right

#use autograd to calculate the gradients
def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),create_graph=True,only_inputs=True)[0]                         
    else:
        return gradients(gradients(u, x), x, order=order - 1)

#calculate the three relative errors
def get_err(net,eps):
    net = net.cpu()
    x0 = torch.linspace(0, 1, 1001)
    x,y = torch.meshgrid(x0,x0)
    x,y = x.reshape(-1,1),y.reshape(-1,1)
    X = torch.cat([x,y],dim=1)
    X.requires_grad_(True)
    u = net(X)
    u_grad = gradients(u,X,order=1)
    ux,uy = u_grad[:,0], u_grad[:,1]
    ux,uy = ux.reshape(-1,1001)[1:-1,1:-1],uy.reshape(-1,1001)[1:-1,1:-1]
    u,ux,uy = u.detach().numpy().flatten(),ux.detach().numpy().flatten(),uy.detach().numpy().flatten()
    u_FDM,ux_FDM,uy_FDM = generate_FDM(eps)
    u_FDM,ux_FDM,uy_FDM = u_FDM.flatten(),ux_FDM.flatten(),uy_FDM.flatten()
    L2_err = np.sqrt(((u - u_FDM) ** 2).sum()) / np.sqrt((u_FDM ** 2).sum())
    H1_err = np.sqrt(((u - u_FDM) ** 2).sum()+((ux - ux_FDM) ** 2).sum() + ((uy - uy_FDM) ** 2).sum()
             ) / np.sqrt((u_FDM ** 2).sum()+(ux_FDM ** 2).sum() + (uy_FDM ** 2).sum())                 
    L_inf_err = np.max(abs(u - u_FDM)) / np.max(abs(u_FDM))
    return L2_err, H1_err, L_inf_err


# the model of PINN
class PINN(torch.nn.Module):
    def __init__(self,input_dim=2,m=32):
        super(PINN, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(m, m),
            torch.nn.Tanh(),
            torch.nn.Linear(m, m),
            torch.nn.Tanh(),
            torch.nn.Linear(m, m),
            torch.nn.Tanh(),
            torch.nn.Linear(m, m),
            torch.nn.Tanh(),
            torch.nn.Linear(m, m),
            torch.nn.Tanh(),
            torch.nn.Linear(m, m),
            torch.nn.Tanh(),
            torch.nn.Linear(m, m),
            torch.nn.Tanh(),
            torch.nn.Linear(m, m),
            torch.nn.Tanh(),
            torch.nn.Linear(m, 1)
        )

    def forward(self, x):
        return self.net(x)



#the training process of PINN
def train(net, epochs, data_iter,batch_size_bd,device, lr, eps,lam_b):
    print('Training on %s' % device)
    optimizer = optims.Adam(net.parameters(), lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    t_all = 0.0
    c1 = 2 * math.pi / eps
    net = net.to(device=device)
    for epoch in range(1, epochs + 1):
        l_r_sum = 0.0
        l_bd_sum = 0.0
        t1 = time.time()
        for x,y in data_iter:
            x = x.to(device=device)
            y = y.to(device=device)
            x.requires_grad_(True)
            y.requires_grad_(True)
            a = (1.5 + torch.sin(c1 * x)) / (1.5 + torch.sin(c1 * y)) + \
                (1.5 + torch.sin(c1 * y)) / (1.5 + torch.cos(c1 * x)) + torch.sin(4 * x ** 2 * y ** 2) + 1
            f = 10 * torch.ones_like(x)
            u = net(torch.cat((x,y),dim=1))
            ux,uy = gradients(u,x,order=1),gradients(u,y,order=1)
            temp = gradients(a*ux,x,order=1)+gradients(a*uy,y,order=1)
            l_r = ((temp-f)**2).mean()
            l_bd = get_bd_loss(net,device,batch_size_bd)
            l = l_r + lam_b * l_bd
            optimizer.zero_grad()
            l.backward(retain_graph=True)
            optimizer.step()
            l_r_sum += l_r.cpu().item()
            l_bd_sum += l_bd.cpu().item()
        t2 = time.time()
        t_all += t2 - t1
        l_sums = l_r_sum  + l_bd_sum
        if (optimizer.param_groups[0]['lr'] > 0.00001):
            scheduler.step()
        print('eps=%.3f,t_all=%.3f,t_epoch=%.3f,epoch %d,l_r:%f,l_bd:%f,l_all:%f'
              % (eps, t_all, t2 - t1, epoch, l_r_sum, l_bd_sum, l_sums))
        if epoch %10==0:
            #torch.save(net.state_dict(), 'model/2d_exam6_%.3f.pth' % (eps))
            L2, H1, L_inf = get_err(net,eps)
            print('Rel. L2:', L2,'Rel. L_inf:', L_inf, 'Rel. H1:', H1 )
            print('--------lr:{0:.6f}'.format(optimizer.param_groups[0]['lr']))
            net = net.to(device)

# show point-wise error |u^S-u^F|
def show_pointwise_error(net,eps,device, dx=0.001):
    x0 = torch.linspace(0, 1, int(1 / dx + 1))
    x, y = torch.meshgrid(x0, x0)
    x, y = x.flatten(), y.flatten()
    X = torch.cat((x.view(-1, 1), y.view(-1, 1)), dim=1)  # [[0,0],[0,0.001],...]
    with torch.no_grad():
        u = net(X)
    u = u.reshape(-1,1001)
    u_FDM,_,_ = generate_FDM(eps)
    err_u = abs(u-u_FDM)
    plt.figure( )
    ax =  plt.subplot(1,1,1)
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    h = plt.imshow(err_u.T, interpolation='nearest', cmap='rainbow',
                   extent=[0, 1, 0, 1],
                   origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    plt.show()

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size
    batch_size_bd = args.batch_size_bd
    epochs = args.epochs
    lr = args.lr
    eps = args.eps
    h = args.h
    data_iter = dataloader(h,batch_size,eps)
    m = args.m
    lam_b = args.lam_b
    net = PINN(input_dim=2,m=m) 
    if args.pretrained:
        if os.path.exists('model/2d_exam6_%.3f.pth' % (eps)):
            net.load_state_dict(torch.load('model/2d_exam6_%.3f.pth' % (eps)))
            print('load the trained model')
        else:
            print('no trained model')
    train(net, epochs, data_iter,batch_size_bd,device, lr, eps,lam_b)
    net = net.cpu()
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameters for PINN: %d" % (total))
    L2, H1, L_inf = get_err(net,eps)
    print('Rel. L2:', L2,'Rel. L_inf:', L_inf, 'Rel. H1:', H1 )
    show_pointwise_error(net,eps,device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--eps', type=float, default=0.5) #[0.5,0.2,0.1,0.05,0.02,0.01]
    parser.add_argument('--m', type=int, default=64) #[64,64,128,128,256,256], the number of hidden layers
    parser.add_argument('--batch_size',type=int, default=1024)
    parser.add_argument('--batch_size_bd',type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--h', type=float, default=0.002)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lam_b', type=float, default=10.0)
    parser.add_argument('--pretrained', type=str, default=True)
    args = parser.parse_args()
    main(args)



"""       params
eps=0.5    29377             L2: 0.011866597479735095 H1: 0.10911698704811248 L_inf: 0.06528397415709067
eps=0.2    29377            L2: 0.6238298231667675 H1: 0.6235284513045328 L_inf: 0.6809156669398957
eps=0.1    116097           L2: 0.8745367095047448 H1: 0.8620112419182806 L_inf: 0.9096070434224083
epss=0.05   116097          L2: 0.9738392906075184 H1: 0.9757944561251705 L_inf: 0.9820454140090963
eps=0.02    461569          L2: 0.9969480621311932 H1: 0.9962055035551565 L_inf: 0.997965772632795
eps=0.01    461569          L2: 1.0066492383692716 H1: 0.9996579529526355 L_inf: 1.0045573830894954

hidden layers :[64,64,128,128,256,256]
"""









