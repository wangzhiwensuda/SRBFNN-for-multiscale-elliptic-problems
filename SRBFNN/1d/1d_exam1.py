import torch
import torch.optim as optims
from torch import nn
import torch.utils.data as Data
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import argparse
import time
import torch.optim.lr_scheduler as lr_scheduler
torch.set_printoptions(precision=8)
torch.manual_seed(1)




# generate training data
def dataloader(num_points, batch_size, eps):
    x = torch.rand(num_points)
    a = 2 + torch.sin(2 * math.pi * x / eps)
    dataset = Data.TensorDataset(x, a)
    data_iter = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_iter

#load the FDM solution
def generate_FDM(eps):
    if os.path.exists('1d_FDM/1d_exam1_%.3f.npy'% (eps)): 
        u_FDM = np.load('1d_FDM/1d_exam1_%.3f.npy' % (eps)).flatten()
        ux_FDM = np.zeros_like(u_FDM)
        ux_FDM[1:-1] = (u_FDM[2:] - u_FDM[:-2]) / 0.0002
        ux_FDM[0] = -1.5 * u_FDM[0] / 0.0001 + 2 * u_FDM[1] / 0.0001 - 0.5 * u_FDM[2] / 0.0001
        ux_FDM[-1] = 1.5 * u_FDM[-1] / 0.0001 - 2 * u_FDM[-2] / 0.0001 + 0.5 * u_FDM[-3] / 0.0001
    else:
        raise ValueError("Run: python 1d_FDM.py, to generate the FDM solution")
    return u_FDM, ux_FDM


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
def get_u(x, c, h, w):  
    c1 = (x.view(-1, 1) - c.view(-1,1,1)) ** 2 
    d2 = (w** 2).view(-1,1,1)   
    r = -torch.matmul(c1,d2)
    m = torch.exp(r) 
    output = torch.matmul(h,m.squeeze(-1))
    return output.flatten()

#The explicit expression of RBFNN's derivatives:
def get_ux(x, c, h, w):
    c1 = (x.view(-1, 1) - c.view(-1,1,1)) 
    d2 = (w** 2).view(-1,1,1) 
    r = -torch.matmul(c1** 2, d2)
    m = -2*torch.matmul(c1,d2)*torch.exp(r)  
    output = torch.matmul(h,m.squeeze(-1))
    return output.flatten()


#calculate the three relative errors
def get_err(net,eps):
    net = net.cpu()
    x = torch.linspace(0, 1, 10001)
    a = 2 + torch.sin(2 * math.pi * x / eps)
    with torch.no_grad():
        u_SRBF = get_u(x, net.center, net.hight, net.width).numpy()
        p = get_u(x, net.center2, net.hight2, net.width2)
        ux_SRBF = (p/a).numpy()
    u,ux = u_SRBF.flatten(),ux_SRBF.flatten()
    u_FDM, ux_FDM = generate_FDM(eps)
    L2_err = np.sqrt(((u - u_FDM) ** 2).sum())/np.sqrt((u_FDM**2).sum())
    H1_err = np.sqrt((((ux - ux_FDM) ** 2).sum()  + ((u - u_FDM) ** 2).sum()))/(np.sqrt((ux_FDM** 2).sum()+ (u_FDM ** 2).sum()))
    L_inf_err = np.max(abs(u - u_FDM))/np.max(abs(u_FDM))
    return L2_err,L_inf_err, H1_err

#calculate the boundary loss
def get_bd_loss(net,device):
    x0 = torch.zeros(1).to(device=device)
    l_a = get_u(x0, net.center, net.hight, net.width) - 1
    l_b = get_u(x0 + 1, net.center, net.hight, net.width) - 1
    l_bd = (l_b ** 2 + l_a ** 2)
    return l_bd

# SRBFNN for 1d
class SRBF(nn.Module):
    """
    hight: The weight of RBFs
    center: The center of RBFs
    width: The shape parameter of RBFs
    """
    def __init__(self, N, eps):
        super(SRBF, self).__init__()
        self.hight = nn.Parameter(torch.rand(N))  
        self.center = nn.Parameter(torch.rand(N, 1))
        self.width = nn.Parameter(torch.rand(N, 1) * (5 / eps))

        self.hight2 = nn.Parameter(torch.rand(N))
        self.center2 = nn.Parameter(torch.rand(N, 1))
        self.width2 = nn.Parameter(torch.rand(N, 1) * (5 / eps))


    def forward(self, x):
        ux = get_ux(x, self.center, self.hight, self.width)
        p = get_u(x, self.center2, self.hight2, self.width2)
        px = get_ux(x, self.center2, self.hight2, self.width2)
        return ux,p, px 


def show_err(net,eps):
    net = net.cpu()
    x = np.linspace(0, 1, 10001)
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示符号
    plt.grid(linestyle="--")
    with torch.no_grad():
        u_SRBF = get_u(torch.Tensor(x), net.center, net.hight, net.width).numpy()
    u_FDM,ux_FDM = generate_FDM(eps)
    plt.plot(x,u_SRBF)
    plt.plot(x,u_FDM)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend(['SRBFNN','FDM'])
    plt.show()
    plt.plot(x,abs(u_SRBF-u_FDM),lw=0.6)
    plt.xlabel('x',fontsize=16)
    plt.ylabel('The absolute error',fontsize=16)
    plt.show()




# 定义训练函数
def train(net,data_iter,eps,device,MaxNiter,SparseNiter,lr,tol1,tol2,Check_iter,lam1,lam2,lam3):
    print('Training on %s' % device)
    net = net.to(device=device)
    optimizer = optims.Adam(net.parameters(), lr)
    t_all,j = 0.0,0
    L_rec = [0.0]
    l_sums = 0.0
    thres = 0.0 
    for Niter in range(1, MaxNiter + 1):
        l_p_sum = 0.0
        l_f_sum = 0.0
        l_bd_sum = 0.0
        t1 = time.time()
        for x, a in data_iter:
            x = x.to(device=device)
            a = a.to(device=device)
            ux,p, px  = net(x)
            l_bd = get_bd_loss(net,device)
            l_f = ((px + 1) ** 2).mean()
            l_p = ((ux - p/a) ** 2).mean()
            l = l_p + lam1*l_f + lam2*l_bd
            if l_sums < thres:
                l = l + lam3 * net.hight.norm(1)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_p_sum += l_p.cpu().item()
            l_f_sum += l_f.cpu().item()
            l_bd_sum += l_bd.cpu().item()
        t2 = time.time()
        t_all += t2-t1
        l_sums = l_p_sum + l_f_sum+l_bd_sum
        if (Niter%300==0)&(optimizer.param_groups[0]['lr']>0.001):
            optimizer.param_groups[0]['lr']=0.1*optimizer.param_groups[0]['lr']
        print('eps=%.3f,t_all=%.3f,t_Niter=%.3f,thres=%.4f,Niter:%d,l_p:%f,l_f:%f,l_bd:%f,l_all:%f' 
            % (eps,t_all,t2-t1, thres, Niter, l_p_sum, l_f_sum,l_bd_sum, l_sums))
        if Niter == SparseNiter: 
            thres = 0.0
        if (Niter % Check_iter == 0):
            L_rec.append(l_sums)
            if thres>0.0:
                net = drop_bf(net,tol2)
                net = net.to(device)
                lr_new = optimizer.param_groups[0]['lr']
                optimizer = optims.Adam(net.parameters(),lr=lr_new)
            if not os.path.exists('model/exam1/'):
                os.makedirs('model/exam1/')
            #torch.save(net.state_dict(), 'model/exam1/1d_exam1_%.3f.pth' % (eps))
            net = net.cpu()
            L2,L_inf,H1 = get_err(net,eps)
            print('L2:', L2,'L_inf:', L_inf,'H1:', H1)
            if (abs(L_rec[-1] - L_rec[-2]) < tol1) & (j==0):
                thres = L_rec[-1] + tol1
                j = j+1
            print('The number of RBFs:{} '.format(net.hight.detach().shape[0]))
            print('The learning rate:{} '.format(optimizer.param_groups[0]['lr']))
            net = net.to(device)


def load_pth(net,eps):
    if os.path.exists('model/exam1/1d_exam1_%.3f.pth' % (eps)):
        print('load the trained model')
        ckpt = torch.load('model/exam1/1d_exam1_%.3f.pth' % (eps))
        net.center = nn.Parameter(ckpt['center'])
        net.hight = nn.Parameter(ckpt['hight'])
        net.width = nn.Parameter(ckpt['width'])
        net.center2 = nn.Parameter(ckpt['center2'])
        net.hight2 = nn.Parameter(ckpt['hight2'])
        net.width2 = nn.Parameter(ckpt['width2'])
    else:
        print('no trained model')
    return net


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size
    N = args.N 
    MaxNiter,SparseNiter = args.MaxNiter,args.SparseNiter
    lr = args.lr
    eps = args.eps
    num_points = args.num_points
    lam1,lam2,lam3 = args.lam1,args.lam2,args.lam3
    tol1,tol2 = args.tol1,args.tol2
    Check_iter = args.Check_iter
    data_iter = dataloader(num_points, batch_size, eps)
    net = SRBF(N,eps)
    if args.pretrained:
        net = load_pth(net,eps)
    #train(net,data_iter,eps,device,MaxNiter,SparseNiter,lr,tol1,tol2,Check_iter,lam1,lam2,lam3)
    print('The number of RBFs in final solution: ',net.hight.shape[0])
    L2,L_inf,H1 = get_err(net,eps)
    print('Rel. L2:', L2,'Rel. L_inf:', L_inf,'Rel. H1:', H1)
    show_err(net,eps)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--eps', type=float, default=0.5)
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--batch_size',type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num_points', type=int, default=10000)
    parser.add_argument('--MaxNiter', type=int, default=3000)
    parser.add_argument('--SparseNiter', type=int, default=2000)
    parser.add_argument('--Check_iter', type=int, default=100)
    parser.add_argument('--pretrained', type=str, default=True)
    parser.add_argument('--lam1', type=float, default=1.0)
    parser.add_argument('--lam2', type=float, default=100.0)
    parser.add_argument('--lam3', type=float, default=0.001)
    parser.add_argument('--tol1', type=float, default=0.001)
    parser.add_argument('--tol2', type=float, default=0.00001)
    args = parser.parse_args()
    main(args)

"""         基函数个数        相对L2误差              相对H1误差             相对L_inf误差
eps=0.5      17               Rel. L2: 3.100091946354364e-05 Rel. L_inf: 8.552424358788451e-05 Rel. H1: 0.0001986455220698837   
eps=0.1      38              Rel. L2: 6.718142144073114e-05 Rel. L_inf: 0.0002348345099692035 Rel. H1: 0.0008559916520817125
eps=0.05     66             Rel. L2: 8.311182670421178e-05 Rel. L_inf: 0.0003291089057338414 Rel. H1: 0.0007847075199158606
eps=0.01    186             Rel. L2: 0.00010860751950087168 Rel. L_inf: 0.0004052113196214109 Rel. H1: 0.0008433103055718821
eps=0.005   367             Rel. L2: 7.48110200262596e-05 Rel. L_inf: 0.0004106747668954035 Rel. H1: 0.0010170854418796596
eps=0.002   750             Rel. L2: 0.0002541406072046563 Rel. L_inf: 0.0003940354571117779 Rel. H1: 0.0018988869053948897
"""






