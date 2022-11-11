

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
import torch.optim.lr_scheduler as lr_scheduler
from mpl_toolkits.axes_grid1 import make_axes_locatable
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


# 将a(x,y)和F(x,y)载入dataloader
def dataloader(dx, batch_size, eps):
    k = 2 * math.pi / eps
    x0 = torch.linspace(0, 1, int(1 / dx + 1))
    x,y,z = torch.meshgrid(x0,x0,x0)
    x,y,z = x.flatten(),y.flatten(),z.flatten()
    X = torch.cat((x.view(-1, 1), y.view(-1, 1), z.view(-1,1)), dim=1)
    a = 2+torch.sin(k*x)*torch.sin(k*y)*torch.sin(k*z)
    #f = get_f(x,y,z,a)
    f = 10*torch.ones_like(x)
    dataset = Data.TensorDataset(X, a, f)
    data_iter = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_iter



# 丢弃h小于阈值的基函数
def drop_bf(net, thres=0.00001):
    net = net.cpu()
    print(f'-----丢弃前基函数个数：{net.hight.shape[0]}')
    # show_u(net.center,net.hight,net.width)
    c, h, w = net.center.detach(), net.hight.detach(), net.width.detach()
    index = torch.where(abs(h) > thres)[0]
    c1 = torch.index_select(c, 0, index)
    h1 = torch.index_select(h, 0, index)
    w1 = torch.index_select(w, 0, index)
    net.center = nn.Parameter(c1)
    net.hight = nn.Parameter(h1)
    net.width = nn.Parameter(w1)
    print(f'丢弃后基函数个数------：{net.hight.shape[0]}')
    # show_u(net.center,net.hight,net.width)
    return net.to(device)


# 定义RBF
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


# 计算边界损失
def get_bound_loss(c, h, w, dx=0.005):  # c:n*2,,,h:n,,,w:n
    x = torch.linspace(0, 1, int(1 / dx) + 1).to(device)
    y = torch.zeros_like(x).to(device)
    z = torch.ones_like(x).to(device)
    X_1 = torch.cat((x.view(-1, 1), y.view(-1, 1),y.view(-1, 1)), dim=1)  # (x,0,0)
    X_2 = torch.cat((x.view(-1, 1), y.view(-1, 1),z.view(-1, 1)), dim=1)  #(x,0,1)
    X_3 = torch.cat((x.view(-1, 1), z.view(-1, 1),y.view(-1, 1)), dim=1)  # (x,1,0)
    X_4 = torch.cat((x.view(-1, 1), z.view(-1, 1),z.view(-1, 1)), dim=1)  #(x,1,1)
    
    Y_1 = torch.cat((y.view(-1, 1), x.view(-1, 1), y.view(-1, 1)), dim=1)  # (0,y,0)
    Y_2 = torch.cat((y.view(-1, 1), x.view(-1, 1), z.view(-1, 1)), dim=1)  # (0,y,1)
    Y_3 = torch.cat((z.view(-1, 1), x.view(-1, 1), y.view(-1, 1)), dim=1)  # (1,y,0)
    Y_4 = torch.cat((z.view(-1, 1), x.view(-1, 1), z.view(-1, 1)), dim=1)  # (1,y,1)
    
    Z_1 = torch.cat((y.view(-1, 1), z.view(-1, 1), x.view(-1, 1)), dim=1)  # (0,1,z)
    Z_2 = torch.cat((y.view(-1, 1), y.view(-1, 1), x.view(-1, 1)), dim=1)  # (0,0,z)
    Z_3 = torch.cat((z.view(-1, 1), y.view(-1, 1), x.view(-1, 1)), dim=1)  # (1,0,z)
    Z_4 = torch.cat((z.view(-1, 1), z.view(-1, 1), x.view(-1, 1)), dim=1)  # (1,1,z)
    bound_loss = ((get_u(X_1, c, h, w) - 0) ** 2).mean() + ((get_u(X_2, c, h, w) - 0) ** 2).mean() + \
                 ((get_u(X_3, c, h, w) - 0) ** 2).mean() + ((get_u(X_4, c, h, w) - 0) ** 2).mean()+ \
                 ((get_u(Y_1, c, h, w) - 0) ** 2).mean() + ((get_u(Y_2, c, h, w) - 0) ** 2).mean() + \
                 ((get_u(Y_3, c, h, w) - 0) ** 2).mean() + ((get_u(Y_4, c, h, w) - 0) ** 2).mean() + \
                 ((get_u(Z_1, c, h, w) - 0) ** 2).mean() + ((get_u(Z_2, c, h, w) - 0) ** 2).mean() + \
                 ((get_u(Z_3, c, h, w) - 0) ** 2).mean() + ((get_u(Z_4, c, h, w) - 0) ** 2).mean()
    return bound_loss













# 定义模型
class SRBF3d(nn.Module):
    def __init__(self, N1):
        super(SRBF3d, self).__init__()
        self.hight = nn.Parameter(torch.rand(N1))  # 高斯分布0,1
        self.center = nn.Parameter(torch.rand(N1, 3))  # [0,1]均匀分布
        self.width = nn.Parameter(5 * torch.rand(N1, 3) / eps)

        self.hight2 = nn.Parameter(torch.rand(N1))  # 高斯分布0,1
        self.center2 = nn.Parameter(torch.rand(N1, 3))  # [0,1]均匀分布
        self.width2 = nn.Parameter(5 * torch.rand(N1, 3) / eps)

        self.hight3 = nn.Parameter(torch.rand(N1))  # 高斯分布0,1
        self.center3 = nn.Parameter(torch.rand(N1, 3))  # [0,1]均匀分布
        self.width3 = nn.Parameter(5 * torch.rand(N1, 3) / eps)

        self.hight4 = nn.Parameter(torch.rand(N1))  # 高斯分布0,1
        self.center4 = nn.Parameter(torch.rand(N1, 3))  # [0,1]均匀分布
        self.width4 = nn.Parameter(5 * torch.rand(N1, 3) / eps)

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


def show_pointwise_err(net):
    x0 = torch.linspace(0,1,201)
    xx,yy = torch.meshgrid(x0,x0)
    xx,yy = xx.reshape(-1,1),yy.reshape(-1,1)
    vals = [0.5]
    for val in vals:
        zz = val*torch.ones_like(xx)
        X = torch.cat([xx,yy,zz],dim=1)
        with torch.no_grad():
            u = get_u(X,net.center,net.hight,net.width)
        u = u.reshape(-1,201)
        plt.figure()
        ax = plt.subplot(1, 1, 1)
        plt.ylabel('z=%.2f'%val, fontsize=14)
        # plt.ylabel('y', fontsize=16)
        h = plt.imshow(abs(u).T, interpolation='nearest', cmap='rainbow',
                       extent=[0, 1, 0, 1],
                       origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(h, cax=cax)
        #plt.savefig('D:/cc/Desktop/tex/SRBF/imgs/3d/exam8/u_%.2f_%.3f.png'%(val,eps))
        plt.show()
        
def cal_u(net):
    x0 = torch.linspace(0,1,101)
    xx,yy,zz = torch.meshgrid(x0,x0,x0)
    xx,yy,zz = xx.reshape(-1,1),yy.reshape(-1,1),zz.reshape(-1,1)
    X = torch.cat([xx,yy,zz],dim=1)
    X_set = Data.TensorDataset(X)
    X_dataloader = Data.DataLoader(dataset=X_set, batch_size=10000, shuffle=False)
    u,ux,uy,uz = [],[],[],[]
    for X_part in X_dataloader:
        X_part[0] = X_part[0].to(device)
        c,h,w = net.center.to(device),net.hight.to(device),net.width.to(device)
        u_part = get_u(X_part[0], c, h, w).cpu().detach().numpy()
        ux_part = get_ux(X_part[0], c, h, w).cpu().detach().numpy()
        uy_part = get_uy(X_part[0], c, h, w).cpu().detach().numpy()
        uz_part = get_uz(X_part[0], c, h, w).cpu().detach().numpy()
        u.extend(u_part)
        ux.extend(ux_part)
        uy.extend(uy_part)
        uz.extend(uz_part)
    ux = np.array(ux).reshape(-1,101, 101)
    uy = np.array(uy).reshape(-1,101, 101)
    uz = np.array(uz).reshape(-1,101, 101)
    u = np.array(u).reshape(-1,101, 101)
    np.savez('output/exam8/sol_%.3f.npz'%eps,u=u,ux=ux,uy=uy,uz=uz)
    print('文件保存完毕')
    return u

def cal_error(net,eps):
    x0 = torch.linspace(0,1,101)
    xx,yy,zz = torch.meshgrid(x0,x0,x0)
    xx,yy,zz = xx.reshape(-1,1),yy.reshape(-1,1),zz.reshape(-1,1)
    X = torch.cat([xx,yy,zz],dim=1)
    X_set = Data.TensorDataset(X)
    X_dataloader = Data.DataLoader(dataset=X_set, batch_size=10000, shuffle=False)
    u= []
    for X_part in X_dataloader:
        X_part[0] = X_part[0].to(device)
        c,h,w = net.center.to(device),net.hight.to(device),net.width.to(device)
        u_part = get_u(X_part[0], c, h, w).cpu().detach().numpy()
        u.extend(u_part)
    u = np.array(u).reshape(-1,101, 101)
    data1 = np.load('output/exam8/sol_%.3f.npz'%eps)
    data2 = np.load('output/exam8/sol_0.050.npz')
    u1_ref = data1['u']
    u2_ref = data2['u']
    L2 = np.linalg.norm(u.flatten()-u2_ref.flatten())
    print('Norm(u-u):',np.linalg.norm(u.flatten()-u1_ref.flatten()),'Norm(u-u^0.05):',L2)
    return L2
    
def show_L2():
    eps_all = [0.5,0.2,0.1]
    eps_inv = [2.0,5.0,10.0]
    errs = []
    plt.grid('--')
    for eps in eps_all:
        net = SRBF3d(1)
        ckpt = torch.load('model/exam8/3d_exam8_%.3f.pth' % (eps), map_location='cpu')
        net = load_ckpt(net, ckpt)
        L2 = cal_error(net,eps)
        errs.append(L2)
    plt.plot(eps_inv,errs,'-*')
    plt.xlabel(r'$1/\varepsilon$',fontsize=16)
    plt.ylabel(r'$\Vert u^{\varepsilon}-u^{0.05}\Vert_2$',fontsize=16)
    plt.show()
    
def show_N():
    eps_all = [0.5,0.2,0.1,0.05]

    N = []
    plt.grid('--')
    for eps in eps_all:
        net = SRBF3d(1)
        ckpt = torch.load('model/exam8/3d_exam8_%.3f.pth' % (eps), map_location='cpu')
        net = load_ckpt(net, ckpt)
        N.append(net.hight.shape[0])
    plt.plot(eps_all,N,'-*')
    plt.xlabel(r'$\varepsilon$',fontsize=16)
    plt.ylabel(r'N',fontsize=16)
    plt.show()
    
# 定义训练函数
def train(net, epochs, data_iter, lr, eps):
    print('Training on %s' % device)
    net = net.to(device=device)
    optimizer = optims.Adam(net.parameters(),lr=lr)
    l_rec = [0.0]
    thres, flag, l_sums = 0.0, True, 0.0
    t_all = 0.0
    for epoch in range(1, epochs + 1):
        l_P_sum = 0.0
        l_Q_sum = 0.0
        l_R_sum = 0.0
        l_f_sum = 0.0
        l_bound_sum = 0.0
        t1 = time.time()
        for x, a, f in data_iter:
            #print(x.shape,a.shape,f.shape)
            x = x.to(device=device)
            a = a.to(device=device)
            f = f.to(device=device)
            ux, uy, uz, P, Px, Q, Qy, R, Rz = net(x)
            #print(ux.shape,Px.shape,R.shape)
            l_P = ((ux - P / a) ** 2).mean()
            l_Q = ((uy - Q / a) ** 2).mean()
            l_R = ((uz - R / a) ** 2).mean()
            l_f = ((Px + Qy + Rz - f) ** 2).mean()
            l_bound = get_bound_loss(net.center, net.hight, net.width)
            l = (l_P + l_Q + l_R) + 0.1 * l_f + 50 * l_bound#+0.0001*net.hight.norm(1)
            if (l_sums < thres):
                print('---加上正则项----')
                l = l + 0.01 * net.hight.norm(1)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_P_sum += l_P.cpu().item()
            l_Q_sum += l_Q.cpu().item()
            l_R_sum += l_R.cpu().item()
            l_f_sum += l_f.cpu().item()
            l_bound_sum += l_bound.cpu().item()
        t2 = time.time()
        t_all += t2-t1
        l_sums = l_P_sum + l_Q_sum + l_f_sum + l_bound_sum
        if optimizer.param_groups[0]['lr']>0.00001:
            scheduler.step()
        print('thres=%.3f,t_all:%.3f,t_epo:%.3f, eps=%.3f,epoch %d,l(P):%f,l(Q):%f,l(R):%f,l(f):%f,l(bound):%f,l(all):%f'
              % (thres,t_all,t2-t1, eps, epoch, l_P_sum, l_Q_sum, l_R_sum, l_f_sum, l_bound_sum, l_sums))
        if epoch>epochs-50:
            thres=0.0
        if epoch %5==0:
            l_rec.append(l_sums)
            if thres>0.0:
                print('----丢弃前基函数个数：%d-------'%net.hight.shape[0])
                net = drop_bf(net)
                print('----丢弃后基函数个数：%d-------'%net.hight.shape[0])
                lr_new = optimizer.param_groups[0]['lr']
                optimizer = optims.Adam(net.parameters(),lr=lr_new)
            if (abs(l_rec[-1]-l_rec[-2])<-0.05)&flag:
                thres = l_sums+0.05
                flag= False
            cal_error(net,eps)
        print('------基函数个数：%d----'%net.hight.shape[0])
        torch.save(net.state_dict(), 'model/exam8/3d_exam8_%.3f.pth' % (eps))
            



def load_ckpt(net, ckpt):
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

    return net





if __name__ == "__main__":
    batch_size = 1024
    n = 1000  #[1000,4000,10000,20000]--->[600,2000,5000,15000]
    epochs = 150
    eps = 0.2
    lr, dx = 0.00001, 0.01  # min(0.1*eps,0.002)

    data_iter = dataloader(dx, batch_size, eps)
    net = SRBF3d(n)
    if os.path.exists('model/exam8/3d_exam8_%.3f.pth' % (eps)):
        ckpt = torch.load('model/exam8/3d_exam8_%.3f.pth' % (eps), map_location='cpu')
        net = load_ckpt(net, ckpt)
        print('加载已训练网络,eps:%.3f,n:%d' % (eps, net.hight.shape[0]))
    print('-----基函数个数：{}-----'.format(net.hight.detach().shape[0]))
    #train(net, epochs, data_iter, lr, eps)
    #cal_u(net)
    #cal_error(net)
    #show_L2()
    show_N()
    #show_pointwise_err(net)
    print(net.hight2.detach().shape[0])


"""
Norm(u^0.5-u^0.05):     74.2    ,,,,,
Norm(u^0.2-u^0.05):     32.6    ,,,,,
Norm(u^0.1-u^0.05):     11.8    ,,,,,

"""











