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
    c1 = 2 * math.pi / eps
    x0 = torch.linspace(0, 1, int(1 / h + 1))
    x,y,z = torch.meshgrid(x0,x0,x0)
    x,y,z = x.flatten(),y.flatten(),z.flatten()
    X = torch.cat((x.view(-1, 1), y.view(-1, 1),z.view(-1,1)), dim=1)
    a = 2 + torch.sin(c1*x)*torch.sin(c1*y)*torch.sin(c1*z)
    f = 10 * torch.ones_like(x)
    dataset = Data.TensorDataset(X, a, f)
    data_iter = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_iter


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
def get_u(X, c, h, w):  # X:m*3,,c:n*3,,h:n,,w:n*3
    x = (X[:, 0]).view(-1, 1)  # m*1
    y = (X[:, 1]).view(-1, 1)  # m*1
    z = (X[:, 2]).view(-1, 1)  # m*1
    x1 = (x - (c[:, 0]).view(-1, 1, 1))  # n*m*1
    y1 = (y - (c[:, 1]).view(-1, 1, 1)) # n*m*1
    z1 = (z - (c[:, 2]).view(-1, 1, 1)) # n*m*1
    d1 = (w[:, 0] ** 2).view(-1, 1, 1)  # n*1*1
    d2 = (w[:, 1] ** 2).view(-1, 1, 1)  # n*1*1
    d3 = (w[:, 2] ** 2).view(-1, 1, 1)
    r = -torch.matmul(x1 ** 2, d1)-torch.matmul(y1 ** 2, d2)-torch.matmul(z1 ** 2, d3)  # n*m*1
    r2 = torch.exp(r)
    output = torch.matmul(r2.squeeze(-1).t(), h)  # n*[n*m] ==> m
    return output  # m

#The RBFNN's derivative about x:
def get_ux(X, c, h, w):  # X:m*3,,c:n*3,,h:n,,w:n*3
    x = (X[:, 0]).view(-1, 1)  # m*1
    y = (X[:, 1]).view(-1, 1)  # m*1
    z = (X[:, 2]).view(-1, 1)
    x1 = (x - (c[:, 0]).view(-1, 1, 1))  # n*m*1
    y1 = (y - (c[:, 1]).view(-1, 1, 1))
    z1 = (z - (c[:, 2]).view(-1, 1, 1))
    d1 = (w[:, 0] ** 2).view(-1, 1, 1)  # n*1*1
    d2 = (w[:, 1] ** 2).view(-1, 1, 1)  # n*1*1
    d3 = (w[:, 2] ** 2).view(-1, 1, 1)
    r = -torch.matmul(x1 ** 2, d1) - torch.matmul(y1 ** 2, d2) - torch.matmul(z1 ** 2, d3)  # n*m*1
    r2 = -2*torch.matmul(x1,d1)*torch.exp(r)
    output = torch.matmul(r2.squeeze(-1).t(), h)  # n*[n*m] ==> m
    return output  # m

#The RBFNN's derivative about y:
def get_uy(X, c, h, w):  # X:m*2,,c:n*2,,h:n,,w:n
    x = (X[:, 0]).view(-1, 1)  # m*1
    y = (X[:, 1]).view(-1, 1)  # m*1
    z = (X[:, 2]).view(-1, 1)
    x1 = (x - (c[:, 0]).view(-1, 1, 1))  # n*m*1
    y1 = (y - (c[:, 1]).view(-1, 1, 1))
    z1 = (z - (c[:, 2]).view(-1, 1, 1))
    d1 = (w[:, 0] ** 2).view(-1, 1, 1)  # n*1*1
    d2 = (w[:, 1] ** 2).view(-1, 1, 1)  # n*1*1
    d3 = (w[:, 2] ** 2).view(-1, 1, 1)
    r = -torch.matmul(x1 ** 2, d1) - torch.matmul(y1 ** 2, d2) - torch.matmul(z1 ** 2, d3)  # n*m*1
    r2 = -2 * torch.matmul(y1, d2) * torch.exp(r)
    output = torch.matmul(r2.squeeze(-1).t(), h)  # n*[n*m] ==> m
    return output  # m

#The RBFNN's derivative about z:
def get_uz(X, c, h, w):  # X:m*2,,c:n*2,,h:n,,w:n
    x = (X[:, 0]).view(-1, 1)  # m*1
    y = (X[:, 1]).view(-1, 1)  # m*1
    z = (X[:, 2]).view(-1, 1)
    x1 = (x - (c[:, 0]).view(-1, 1, 1))  # n*m*1
    y1 = (y - (c[:, 1]).view(-1, 1, 1))
    z1 = (z - (c[:, 2]).view(-1, 1, 1))
    d1 = (w[:, 0] ** 2).view(-1, 1, 1)  # n*1*1
    d2 = (w[:, 1] ** 2).view(-1, 1, 1)  # n*1*1
    d3 = (w[:, 2] ** 2).view(-1, 1, 1)
    r = -torch.matmul(x1 ** 2, d1) - torch.matmul(y1 ** 2, d2) - torch.matmul(z1 ** 2, d3)  # n*m*1
    r2 = -2 * torch.matmul(z1, d3) * torch.exp(r)
    output = torch.matmul(r2.squeeze(-1).t(), h)  # n*[n*m] ==> m
    return output  # m


#calculate the boundary loss
def get_bound_loss(net, device, batch_size_bd=200):
    coords = torch.linspace(0, 1, batch_size_bd).to(device)
    
    y1, z1 = torch.meshgrid(coords, coords, indexing='ij')
    x1 = torch.zeros_like(y1)
    X_0 = torch.cat((x1.reshape(-1,1), y1.reshape(-1,1), z1.reshape(-1,1)), dim=1)
    
    x2 = torch.ones_like(y1)
    X_1 = torch.cat((x2.reshape(-1,1), y1.reshape(-1,1), z1.reshape(-1,1)), dim=1)
    
    x3, z3 = torch.meshgrid(coords, coords, indexing='ij')
    y3 = torch.zeros_like(x3)
    Y_0 = torch.cat((x3.reshape(-1,1), y3.reshape(-1,1), z3.reshape(-1,1)), dim=1)
    
    y4 = torch.ones_like(x3)
    Y_1 = torch.cat((x3.reshape(-1,1), y4.reshape(-1,1), z3.reshape(-1,1)), dim=1)
    
    x5, y5 = torch.meshgrid(coords, coords, indexing='ij')
    z5 = torch.zeros_like(x5)
    Z_0 = torch.cat((x5.reshape(-1,1), y5.reshape(-1,1), z5.reshape(-1,1)), dim=1)
    
    z6 = torch.ones_like(x5)
    Z_1 = torch.cat((x5.reshape(-1,1), y5.reshape(-1,1), z6.reshape(-1,1)), dim=1)
    
    c, h, w = net.center, net.hight, net.width
    bound_loss = ((get_u(X_0, c, h, w) ** 2).mean()
    bound_loss += ((get_u(X_1, c, h, w) ** 2).mean()
    bound_loss += ((get_u(Y_0, c, h, w) ** 2).mean()
    bound_loss += ((get_u(Y_1, c, h, w) ** 2).mean()
    bound_loss += ((get_u(Z_0, c, h, w) ** 2).mean()
    bound_loss += ((get_u(Z_1, c, h, w) ** 2).mean()
    
    return bound_loss

# SRBFNN for 3d
class SRBF3d(nn.Module):
    def __init__(self, N,eps):
        super(SRBF3d, self).__init__()
        self.hight = nn.Parameter(torch.rand(N)) 
        self.center = nn.Parameter(torch.rand(N, 3))  
        self.width = nn.Parameter(5 * torch.rand(N, 3) / eps)

        self.hight2 = nn.Parameter(torch.rand(N))  
        self.center2 = nn.Parameter(torch.rand(N, 3))  
        self.width2 = nn.Parameter(5 * torch.rand(N, 3) / eps)

        self.hight3 = nn.Parameter(torch.rand(N))  
        self.center3 = nn.Parameter(torch.rand(N, 3))  
        self.width3 = nn.Parameter(5 * torch.rand(N, 3) / eps)

        self.hight4 = nn.Parameter(torch.rand(N))  
        self.center4 = nn.Parameter(torch.rand(N, 3)) 
        self.width4 = nn.Parameter(5 * torch.rand(N, 3) / eps)

    def forward(self, x):
        ux = get_ux(x, self.center, self.hight, self.width)
        uy = get_uy(x, self.center, self.hight, self.width)
        uz = get_uz(x, self.center, self.hight, self.width)

        P = get_u(x, self.center2, self.hight2, self.width2)
        Px = get_ux(x, self.center2, self.hight2, self.width2)

        Q = get_u(x, self.center3, self.hight3, self.width3)
        Qy = get_uy(x, self.center3, self.hight3, self.width3)

        R = get_u(x, self.center4, self.hight4, self.width4)
        Rz = get_uz(x, self.center4, self.hight4, self.width4)
        return ux, uy, uz, P, Px, Q, Qy, R, Rz

     

#calculate the solution at eps     
def cal_u(net,eps,device):
    x0 = torch.linspace(0,1,101)
    x,y,z = torch.meshgrid(x0,x0,x0)
    x,y,z = x.reshape(-1,1),y.reshape(-1,1),z.reshape(-1,1)
    X = torch.cat([x,y,z],dim=1)
    X_set = Data.TensorDataset(X)
    X_dataloader = Data.DataLoader(dataset=X_set, batch_size=10000, shuffle=False)
    u = []
    net = net.to(device)
    with torch.no_grad():
        for X_part in X_dataloader:
            X_part[0] = X_part[0].to(device)
            u_part = get_u(X_part[0], net.center, net.hight, net.width).cpu().numpy()
            u.extend(u_part)
    u = np.array(u).reshape(-1,101, 101)
    np.savez('output/exam8/solution_%.3f.npz'%eps,u=u)
    return u


#calculate the error norm between the SRBFNN solution at eps=0.05 and the solutions at larger scales
def cal_error_norm(net,eps,device):
    if os.path.exists('output/exam8/solution_0.050.npz'):
        u_ref = np.load('output/exam8/solution_0.050.npz')['u']
    else:
        raise ValueError("Generate the reference solution, i.e., the SRBF solution at eps=0.05")
    u = cal_u(net,eps,device)
    L2_norm = np.linalg.norm(u_ref.flatten()-u.flatten())
    return L2_norm
    
    
#show the error norm 
def show_L2N(device):
    eps_all = [0.5,0.2,0.1,0.05]
    eps_inv = [2.0,5.0,10.0,20.0]
    errs_norm,N = [],[]
    plt.grid('--')
    for eps in eps_all:
        net = SRBF3d(1,0.1)
        net = load_pth(net,eps)
        L2 = cal_error_norm(net,eps,device)
        errs_norm.append(L2)
        N.append(net.hight.shape[0])
    plt.plot(eps_inv,errs_norm,'-*')
    plt.xlabel(r'$1/\varepsilon$',fontsize=16)
    plt.ylabel(r'$\Vert u^{\varepsilon}-u^{0.05}\Vert_2$',fontsize=16)
    plt.show()
    plt.grid('--')
    plt.plot(np.log(eps_all),np.log(N),'-*')
    plt.xlabel(r'ln($\varepsilon$)',fontsize=16)
    plt.ylabel(r'ln($N$)',fontsize=16)
    plt.show()
    
#show the slice of the SRBFNN solution at z=0.5
def show_slice_solution(net):
    net = net.cpu()
    x0 = torch.linspace(0,1,201)
    x,y = torch.meshgrid(x0,x0)
    x,y = x.reshape(-1,1),y.reshape(-1,1)
    z = 0.5*torch.ones_like(x)
    X = torch.cat([x,y,z],dim=1)
    with torch.no_grad():
        u = get_u(X,net.center,net.hight,net.width)
    u = u.reshape(-1,201)
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.ylabel('z=0.5', fontsize=14)
    h = plt.imshow(u, interpolation='nearest', cmap='rainbow',
                   extent=[0, 1, 0, 1],
                   origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    plt.show()  
    
# The training process of SRBFNN 
def train(net,data_iter,batch_size_bd,eps,device,MaxNiter,SparseNiter,lr,tol1,tol2,Check_iter,lam1,lam2,lam3):
    print('Training on %s' % device)
    optimizer = optims.Adam(net.parameters(), lr)
    t_all,j = 0.0,0
    L_rec = [0.0]
    l_sums = 0.0
    thres = 0.0 
    net = net.to(device=device)
    for Niter in range(1, MaxNiter + 1):
        l_P_sum = 0.0
        l_Q_sum = 0.0
        l_R_sum = 0.0
        l_f_sum = 0.0
        l_bd_sum = 0.0
        t1 = time.time()
        for x, a, f in data_iter:
            x = x.to(device=device)
            a = a.to(device=device)
            f = f.to(device=device)
            ux, uy,uz, P, Px, Q, Qy, R, Rz  = net(x)
            l_P = ((ux - P/a ) ** 2).mean()
            l_Q = ((uy - Q/a) ** 2).mean()
            l_R = ((uz - R/a) ** 2).mean()
            l_f = ((Px + Qy+Rz + f) ** 2).mean()
            l_bound = get_bound_loss(net,device, batch_size_bd)
            l = l_P + l_Q+l_R + lam1 * l_f + lam2 * l_bound
            if l_sums < thres:
                l = l + lam3 * net.hight.norm(1)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_P_sum += l_P.cpu().item()
            l_Q_sum += l_Q.cpu().item()
            l_R_sum += l_R.cpu().item()
            l_f_sum += l_f.cpu().item()
            l_bd_sum += l_bound.cpu().item()
        t2 = time.time()
        t_all += t2-t1
        l_sums = l_P_sum +l_Q_sum+l_R_sum+ l_f_sum+l_bd_sum
        if (Niter%30==0)&(optimizer.param_groups[0]['lr']>0.0001):
            optimizer.param_groups[0]['lr']=0.1*optimizer.param_groups[0]['lr']
        print('eps=%.3f,t_all=%.3f,t_Niter=%.3f,thres=%.4f,Niter:%d,l_P:%f,l_Q:%f,l_R:%f,l_f:%f,l_bd:%f,l_all:%f' 
            % (eps,t_all,t2-t1, thres, Niter, l_P_sum,l_Q_sum,l_R_sum, l_f_sum,l_bd_sum, l_sums))
        if Niter == SparseNiter: 
            thres = 0.0
        if (Niter % Check_iter == 0):
            L_rec.append(l_sums)
            if thres>0.0:
                net = drop_bf(net,tol2)
                net = net.to(device)
                lr_new = optimizer.param_groups[0]['lr']
                optimizer = optims.Adam(net.parameters(),lr=lr_new)
            if not os.path.exists('model/exam8/'):
                os.makedirs('model/exam8/')
            #torch.save(net.state_dict(), 'model/exam8/3d_exam8_%.3f.pth' % (eps))
            if (abs(L_rec[-1] - L_rec[-2]) < tol1) & (j==0):
                thres = L_rec[-1] + tol1
                j = j+1
            print('The number of RBFs:{} '.format(net.hight.detach().shape[0]))
            print('The learning rate:{} '.format(optimizer.param_groups[0]['lr']))
            L2norm = cal_error_norm(net,eps,device)
            print('L2 error norm:', L2norm)
            



#load the trained model
def load_pth(net,eps):
    if os.path.exists('model/exam8/3d_exam8_%.3f.pth' % (eps)):
        print('load the trained model')
        ckpt = torch.load('model/exam8/3d_exam8_%.3f.pth' % (eps))
        net.center = nn.Parameter(ckpt['center'])
        net.hight = nn.Parameter(ckpt['hight'])
        net.width = nn.Parameter(ckpt['width'])
        net.center2 = nn.Parameter(ckpt['center2'])
        net.hight2 = nn.Parameter(ckpt['hight2'])
        net.width2 = nn.Parameter(ckpt['width2'])
        net.center3 = nn.Parameter(ckpt['center3'])
        net.hight3 = nn.Parameter(ckpt['hight3'])
        net.width3 = nn.Parameter(ckpt['width3'])
        net.center4 = nn.Parameter(ckpt['center4'])
        net.hight4 = nn.Parameter(ckpt['hight4'])
        net.width4 = nn.Parameter(ckpt['width4'])
    else:
        print('no trained model')
    return net

        



def main(args):
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
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
    net = SRBF3d(N,eps)
    if args.pretrained:
        net = load_pth(net,eps)
    train(net,data_iter,batch_size_bd,eps,device,MaxNiter,SparseNiter,lr,tol1,tol2,Check_iter,lam1,lam2,lam3)
    print('The number of RBFs in final solution: ',net.hight.shape[0])
    show_slice_solution(net)
    #show_L2N(device)

   

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--eps', type=float, default=0.2) #[0.5,0.2,0.1,0.05]
    parser.add_argument('--N', type=int, default=10000) #[1000,2000,5000,10000]
    parser.add_argument('--pretrained', type=str, default=True)
    parser.add_argument('--batch_size',type=int, default=1024)
    parser.add_argument('--batch_size_bd',type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--h', type=float, default=0.01)
    parser.add_argument('--MaxNiter', type=int, default=150)
    parser.add_argument('--SparseNiter', type=int, default=120)
    parser.add_argument('--Check_iter', type=int, default=10)
    parser.add_argument('--lam1', type=float, default=0.1)
    parser.add_argument('--lam2', type=float, default=50.0)
    parser.add_argument('--lam3', type=float, default=0.001)
    parser.add_argument('--tol1', type=float, default=0.05)
    parser.add_argument('--tol2', type=float, default=0.00001)
    args = parser.parse_args()
    main(args)


"""
Norm(u^0.5-u^0.05):     74.2    ,,,,,
Norm(u^0.2-u^0.05):     32.6    ,,,,,
Norm(u^0.1-u^0.05):     11.8    ,,,,,

"""











