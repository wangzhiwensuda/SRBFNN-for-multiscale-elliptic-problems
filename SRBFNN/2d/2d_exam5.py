import torch
import torch.optim as optims
from torch import nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import time
import math
import argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable



# generate training data
def dataloader(h, batch_size, eps):
    x0 = torch.linspace(0, 1, int(1 / h + 1))
    x,y = torch.meshgrid(x0,x0)
    x,y = x.flatten(),y.flatten()
    X = torch.cat((x.view(-1, 1), y.view(-1, 1)), dim=1)
    a = 2 + torch.sin(2 * math.pi * (x + y) / eps)
    f = 1 * torch.ones_like(x)
    dataset = Data.TensorDataset(X, a, f)
    data_iter = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_iter


#load the FDM solution
def generate_FDM(eps):
    if os.path.exists('2d_FDM/2d_exam5_FDM_%.3f.npy'% (eps)): 
        u_FDM = np.load('2d_FDM/2d_exam5_FDM_%.3f.npy' % (eps))[::2,::2]
        ux_FDM = np.load('2d_FDM/2d_exam5_FDM_x_%.3f.npy' % (eps))[::2,::2][1:-1,1:-1]
        uy_FDM = np.load('2d_FDM/2d_exam5_FDM_y_%.3f.npy' % (eps))[::2,::2][1:-1,1:-1]
    else:
        raise ValueError("Run: python 2d_exam5_FDM.py, to generate the FDM solution")
    return u_FDM, ux_FDM,uy_FDM
    

#delete the RBFs whose absolute value of weight is less than tol2
def drop_bf(net,tol2=0.00001):
    net = net.cpu()
    print(f'The number of RBFs before discarding：{net.hight.shape[0]}')
    c, h, w = net.center.detach(), net.hight.detach(), net.width.detach()
    index = torch.where(abs(h) > tol2)[0]
    c1 = torch.index_select(c, 0, index)
    h1 = torch.index_select(h, 0, index)
    w1 = torch.index_select(w, 0, index)
    net.center = nn.Parameter(c1)
    net.hight = nn.Parameter(h1)
    net.width = nn.Parameter(w1)
    print(f'The number of RBFs after discarding：{net.hight.shape[0]}')
    return net


# RBFNN, x is input, c is the center of RBFs, h is the weight of RBFs, w is the shape parameter (width) of RBFs.
def get_u(X, c, h, w):  # X:m*2,,c:n*2,,h:n,,w:n*2
    x = (X[:, 0]).view(-1, 1)  # m*1
    y = (X[:, 1]).view(-1, 1)  # m*1
    x1 = (x - (c[:, 0]).view(-1, 1, 1))  # n*m*1
    y1 = (y - (c[:, 1]).view(-1, 1, 1))
    d1 = (w[:, 0] ** 2).view(-1, 1, 1)  # n*1*1
    d2 = (w[:, 1] ** 2).view(-1, 1, 1)  # n*1*1
    r = -torch.matmul(x1 ** 2, d1)-torch.matmul(y1 ** 2, d2)  # n*m*1
    r2 = torch.exp(r)
    output = torch.matmul(r2.squeeze(-1).t(), h)  # n*[n*m] ==> m
    return output  # m

#The RBFNN's derivative about x:
def get_ux(X, c, h, w):  # X:m*2,,c:n*2,,h:n,,w:n
    x = (X[:, 0]).view(-1, 1)  # m*1
    y = (X[:, 1]).view(-1, 1)  # m*1
    x1 = (x - (c[:, 0]).view(-1, 1, 1))  # n*m*1
    y1 = (y - (c[:, 1]).view(-1, 1, 1))
    d1 = (w[:, 0] ** 2).view(-1, 1, 1)  # n*1*1
    d2 = (w[:, 1] ** 2).view(-1, 1, 1)  # n*1*1
    r = -torch.matmul(x1 ** 2, d1)-torch.matmul(y1 ** 2, d2)  # n*m*1
    r1 = -2*torch.matmul(x1,d1)*torch.exp(r)
    output = torch.matmul(r1.squeeze(-1).t(), h)  # n*[n*m] ==> m
    return output  # m

#The RBFNN's derivative about y:
def get_uy(X, c, h, w):  # X:m*2,,c:n*2,,h:n,,w:n
    x = (X[:, 0]).view(-1, 1)  # m*1
    y = (X[:, 1]).view(-1, 1)  # m*1
    x1 = (x - (c[:, 0]).view(-1, 1, 1))  # n*m*1
    y1 = (y - (c[:, 1]).view(-1, 1, 1))
    d1 = (w[:, 0] ** 2).view(-1, 1, 1)  # n*1*1
    d2 = (w[:, 1] ** 2).view(-1, 1, 1)  # n*1*1
    r = -torch.matmul(x1 ** 2, d1) - torch.matmul(y1 ** 2, d2)  # n*m*1
    r1 = -2 * torch.matmul(y1, d2) * torch.exp(r)
    output = torch.matmul(r1.squeeze(-1).t(), h)  # n*[n*m] ==> m
    return output  # m


#calculate the boundary loss
def get_bound_loss(net,device, batch_size_bd=512):  
    x = torch.linspace(0, 1, batch_size_bd).to(device)
    y = torch.zeros_like(x).to(device)
    X_1 = torch.cat((x.view(-1, 1), y.view(-1, 1)), dim=1)  # (x,0)
    X_2 = torch.cat(((y + 1).view(-1, 1), x.view(-1, 1)), dim=1)  # (1,y)
    X_3 = torch.cat((x.view(-1, 1), (y+1).view(-1, 1)), dim=1)  # (x,1)
    X_4 = torch.cat((y.view(-1, 1), x.view(-1, 1)), dim=1)  # (0,y)
    bound_loss = ((get_u(X_1,net.center,net.hight,net.width) - 1) ** 2).mean() + ((get_u(X_2,net.center, net.hight, net.width) - 1) ** 2).mean() + \
                 ((get_u(X_3,net.center,net.hight,net.width) - 1) ** 2).mean() + ((get_u(X_4,net.center, net.hight, net.width) - 1) ** 2).mean()
    return bound_loss



# SRBFNN for 2d
class SRBF2d(nn.Module):
    def __init__(self, N,eps):
        super(SRBF2d, self).__init__()
        self.hight = nn.Parameter(torch.rand(N)) 
        self.center = nn.Parameter(torch.rand(N, 2))  
        self.width = nn.Parameter(5 * torch.rand(N, 2) / eps)

        self.hight2 = nn.Parameter(torch.rand(N)) 
        self.center2 = nn.Parameter(torch.rand(N, 2))  
        self.width2 = nn.Parameter(5 * torch.rand(N, 2) / eps)

        self.hight3 = nn.Parameter(torch.rand(N))  
        self.center3 = nn.Parameter(torch.rand(N, 2))  
        self.width3 = nn.Parameter(5 * torch.rand(N, 2) / eps)

    def forward(self, x):
        ux = get_ux(x, self.center, self.hight, self.width)
        uy = get_uy(x, self.center, self.hight, self.width)

        P = get_u(x, self.center2, self.hight2, self.width2)
        Px = get_ux(x, self.center2, self.hight2, self.width2)

        Q = get_u(x, self.center3, self.hight3, self.width3)
        Qy = get_uy(x, self.center3, self.hight3, self.width3)
        return ux, uy, P, Px, Q, Qy


#calculate the three relative errors
def get_err(net,eps,device,dx = 0.001):
    x0 = torch.linspace(0, 1, int(1 / dx + 1))
    x,y = torch.meshgrid(x0,x0)
    x,y = x.flatten(),y.flatten()
    X = torch.cat((x.view(-1, 1), y.view(-1, 1)), dim=1)  # [[0,0],[0,0.001],...]
    X_set = Data.TensorDataset(X)
    X_dataloader = Data.DataLoader(dataset=X_set, batch_size=10000, shuffle=False)
    u,ux,uy = [],[],[]
    net = net.to(device)
    with torch.no_grad():
        for X_part in X_dataloader:
            X_part[0] = X_part[0].to(device)
            u_part = get_u(X_part[0], net.center, net.hight, net.width).cpu().numpy()
            P_part = get_u(X_part[0], net.center2, net.hight2, net.width2)
            Q_part = get_u(X_part[0], net.center3, net.hight3, net.width3)
            a = 2 + torch.sin(2 * math.pi * (X_part[0][:,0] + X_part[0][:,1]) / eps)
            u.extend(u_part)
            ux.extend((P_part/a).cpu().numpy())
            uy.extend((Q_part/a).cpu().numpy())
    u = np.array(u)
    ux = np.array(ux).reshape(-1, int(1 / dx + 1))[1:-1,1:-1]
    uy = np.array(uy).reshape(-1, int(1 / dx + 1))[1:-1,1:-1]
    u,ux,uy = u.flatten(),ux.flatten(),uy.flatten()
    u_FDM, ux_FDM,uy_FDM = generate_FDM(eps)
    u_FDM, ux_FDM,uy_FDM= u_FDM.flatten(),ux_FDM.flatten(),uy_FDM.flatten()
    L2_err = np.sqrt(((u - u_FDM) ** 2).sum()) / np.sqrt((u_FDM ** 2).sum())
    H1_err = np.sqrt(((u - u_FDM) ** 2).sum()+((ux - ux_FDM) ** 2).sum() + ((uy - uy_FDM) ** 2).sum())/np.sqrt((u_FDM ** 2).sum()+(ux_FDM ** 2).sum() + (uy_FDM ** 2).sum())                
    L_inf_err = np.max(abs(u - u_FDM)) / np.max(abs(u_FDM))
    return L2_err,L_inf_err,H1_err




# The training process of SRBFNN 
def train(net,data_iter,batch_size_bd,eps,device,MaxNiter,SparseNiter,lr,tol1,tol2,Check_iter,lam1,lam2,lam3):
    print('Training on %s' % device)
    optimizer = optims.Adam(net.parameters(), lr)
    t_all,j = 0.0,0
    L_rec = [0.0]
    l_sums = 0.0
    thres = 0.0 
    err_L2_rec, err_H1_rec, err_Li_rec, Niter_rec, N_rec = [], [], [], [], []
    L2, H1, L_inf = get_err(net,eps,device)
    err_L2_rec.append(L2)
    err_H1_rec.append(H1)
    err_Li_rec.append(L_inf)
    Niter_rec.append(0)
    N_rec.append(net.hight.detach().shape[0])
    net = net.to(device=device)
    for Niter in range(1, MaxNiter + 1):
        l_P_sum = 0.0
        l_Q_sum = 0.0
        l_f_sum = 0.0
        l_bd_sum = 0.0
        t1 = time.time()
        for x, a, f in data_iter:
            x = x.to(device=device)
            a = a.to(device=device)
            f = f.to(device=device)
            ux, uy, P, Px, Q, Qy  = net(x)
            l_P = ((ux - P/a ) ** 2).mean()
            l_Q = ((uy - Q/a) ** 2).mean()
            l_f = ((Px + Qy - f) ** 2).mean()
            l_bound = get_bound_loss(net,device, batch_size_bd)
            l = l_P + l_Q + lam1 * l_f + lam2 * l_bound
            if l_sums < thres:
                l = l + lam3 * net.hight.norm(1)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_P_sum += l_P.cpu().item()
            l_Q_sum += l_Q.cpu().item()
            l_f_sum += l_f.cpu().item()
            l_bd_sum += l_bound.cpu().item()
        t2 = time.time()
        t_all += t2-t1
        l_sums = l_P_sum +l_Q_sum+ l_f_sum+l_bd_sum
        if (Niter%40==0)&(optimizer.param_groups[0]['lr']>0.0001):
            optimizer.param_groups[0]['lr']=0.1*optimizer.param_groups[0]['lr']
        print('eps=%.3f,t_all=%.3f,t_Niter=%.3f,thres=%.4f,Niter:%d,l_P:%f,l_Q:%f,l_f:%f,l_bd:%f,l_all:%f' 
            % (eps,t_all,t2-t1, thres, Niter, l_P_sum,l_Q_sum, l_f_sum,l_bd_sum, l_sums))
        if Niter == SparseNiter: 
            thres = 0.0
        if (Niter % Check_iter == 0):
            L_rec.append(l_sums)
            if thres>0.0:
                net = drop_bf(net,tol2)
                net = net.to(device)
                lr_new = optimizer.param_groups[0]['lr']
                optimizer = optims.Adam(net.parameters(),lr=lr_new)
            if not os.path.exists('model/exam5/'):
                os.makedirs('model/exam5/')
            #torch.save(net.state_dict(), 'model/exam5/2d_exam5_%.3f.pth' % (eps))
            if (abs(L_rec[-1] - L_rec[-2]) < tol1) & (j==0):
                thres = L_rec[-1] + tol1
                j = j+1
            print('The number of RBFs:{} '.format(net.hight.detach().shape[0]))
            print('The learning rate:{} '.format(optimizer.param_groups[0]['lr']))
        if Niter % 5 == 0:
            L2, L_inf, H1 = get_err(net,eps,device)
            print('L2:', L2,'L_inf:', L_inf,'H1:', H1)
            err_L2_rec.append(L2)
            err_H1_rec.append(H1)
            err_Li_rec.append(L_inf)
            Niter_rec.append(Niter)
            N_rec.append(net.hight.detach().shape[0])
            if not os.path.exists('output/exam5/'):
                os.makedirs('output/exam5/')
            np.savez('output/exam5/err_rec_%.3f.npz'%eps, H1=err_H1_rec, L2=err_L2_rec, L_inf=err_Li_rec, Niter_rec=Niter_rec,N=N_rec)

#load the trained model
def load_pth(net,eps):
    if os.path.exists('model/exam5/2d_exam5_%.3f.pth' % (eps)):
        print('load the trained model')
        ckpt = torch.load('model/exam5/2d_exam5_%.3f.pth' % (eps))
        net.center = nn.Parameter(ckpt['center'])
        net.hight = nn.Parameter(ckpt['hight'])
        net.width = nn.Parameter(ckpt['width'])
        net.center2 = nn.Parameter(ckpt['center2'])
        net.hight2 = nn.Parameter(ckpt['hight2'])
        net.width2 = nn.Parameter(ckpt['width2'])
        net.center3 = nn.Parameter(ckpt['center3'])
        net.hight3 = nn.Parameter(ckpt['hight3'])
        net.width3 = nn.Parameter(ckpt['width3'])
    else:
        print('no trained model')
    return net


#show the training process of three relative errors and the number of RBFs
def show_rec_err(eps):
    if not os.path.exists('output/exam5/err_rec_%.3f.npz'%eps): 
        raise ValueError("No output file for recording errors!")
    else:
        data = np.load('output/exam5/err_rec_%.3f.npz'%eps)
        err_H1 = data['H1']
        err_L2 = data['L2']
        err_Li = data['L_inf']
        Niter_rec = data['Niter_rec']
        N_rec = data['N']
        plt.grid(linestyle='--')
        plt.plot(Niter_rec, err_H1)
        plt.plot(Niter_rec, err_L2)
        plt.plot(Niter_rec, err_Li)
        plt.semilogy()
        plt.legend(['$H_1$', '$L_2$', r'$L_\infty$'])
        plt.xlabel('Iter', fontsize=16)
        plt.ylabel('Relative error', fontsize=16)
        plt.show()
        plt.grid(linestyle='--')
        plt.plot(Niter_rec,N_rec)
        plt.xlabel('Iter', fontsize=16)
        plt.ylabel('the number of RBFs', fontsize=16)
        plt.show()


# show point-wise error |u^S-u^F|
def show_pointwise_error(net,eps,device, dx=0.001):
    x0 = torch.linspace(0, 1, int(1 / dx + 1))
    x, y = torch.meshgrid(x0, x0)
    x, y = x.flatten(), y.flatten()
    X = torch.cat((x.view(-1, 1), y.view(-1, 1)), dim=1)  # [[0,0],[0,0.001],...]
    X_set = Data.TensorDataset(X)
    X_dataloader = Data.DataLoader(dataset=X_set, batch_size=10000, shuffle=False)
    u = []
    net = net.to(device)
    with torch.no_grad():
        for X_part in X_dataloader:
            X_part[0] = X_part[0].to(device)
            u_part = get_u(X_part[0], net.center, net.hight, net.width).cpu().numpy()
            u.extend(u_part)
    u = np.array(u).reshape(-1, int(1 / dx + 1))
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
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size
    batch_size_bd = args.batch_size_bd
    N = args.N 
    MaxNiter,SparseNiter = args.MaxNiter,args.SparseNiter
    lr = args.lr
    eps = args.eps
    h = args.h
    lam1,lam2,lam3 = args.lam1,args.lam2,args.lam3
    tol1,tol2 = args.tol1,args.tol2
    Check_iter = args.Check_iter
    data_iter = dataloader(h, batch_size, eps)
    net = SRBF2d(N,eps)
    if args.pretrained:
        net = load_pth(net,eps)
    #train(net,data_iter,batch_size_bd,eps,device,MaxNiter,SparseNiter,lr,tol1,tol2,Check_iter,lam1,lam2,lam3)
    print('The number of RBFs in final solution: ',net.hight.shape[0])
    L2,L_inf,H1 = get_err(net,eps,device)
    print('Rel. L2:', L2,'Rel. L_inf:', L_inf,'Rel. H1:', H1)
    show_rec_err(eps)
    show_pointwise_error(net,eps,device)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--eps', type=float, default=0.01) #[0.5,0.2,0.1,0.05,0.02,0.01]
    parser.add_argument('--N', type=int, default=30000) #[1000,1000,2000,5000,15000,30000]
    parser.add_argument('--pretrained', type=str, default=True)
    parser.add_argument('--batch_size',type=int, default=1024)
    parser.add_argument('--batch_size_bd',type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--h', type=float, default=0.002)
    parser.add_argument('--MaxNiter', type=int, default=300)
    parser.add_argument('--SparseNiter', type=int, default=250)
    parser.add_argument('--Check_iter', type=int, default=10)
    parser.add_argument('--lam1', type=float, default=0.004)
    parser.add_argument('--lam2', type=float, default=20.0)
    parser.add_argument('--lam3', type=float, default=0.001)
    parser.add_argument('--tol1', type=float, default=0.1)
    parser.add_argument('--tol2', type=float, default=0.00001)
    args = parser.parse_args()
    main(args)

"""         N     
eps=0.5      42       Rel. L2: 0.000479698649903694 Rel. L_inf: 0.0015966252173473805 Rel. H1: 0.00544183319071744
eps=0.2      146      Rel. L2: 0.0006077568411515522 Rel. L_inf: 0.0013279445829436076 Rel. H1: 0.017210464577106774
eps=0.1     531       Rel. L2: 0.000316669867277307 Rel. L_inf: 0.0014503002166748047 Rel. H1: 0.017718144815479238
eps=0.05    2121      Rel. L2: 0.0003563770971345979 Rel. L_inf: 0.001582503318786621 Rel. H1: 0.026837674168042558
eps=0.02   6272       Rel. L2: 0.0009639546803911716 Rel. L_inf: 0.0018618106830079473 Rel. H1: 0.03189016603026341
eps=0.01   14164      Rel. L2: 0.0007718959488616287 Rel. L_inf: 0.0017931480939576971 Rel. H1: 0.02069962582069043
"""



