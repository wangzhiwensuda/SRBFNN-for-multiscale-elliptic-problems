"""
phi(X):e^{-d1^2(x-c1)^2-d2^2(y-c2)^2}
phix(X):-2d1^2(x-c1)e^{-d1^2(x-c1)^2-d2^2(y-c2)^2}
phiy(X):-2d2^2(y-c2)e^{-d1^2(x-c1)^2-d2^2(y-c2)^2}
"""

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

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


# 将a(x,y)和F(x,y)载入dataloader
def dataloader(dx, batch_size, eps):
    x0 = torch.linspace(0, 1, int(1 / dx + 1))
    x,y = torch.meshgrid(x0,x0)
    x,y = x.flatten(),y.flatten()
    X = torch.cat((x.view(-1, 1), y.view(-1, 1)), dim=1)
    a = 2 + torch.sin(2 * math.pi * (x + y) / eps)
    f = 1 * torch.ones_like(x)
    dataset = Data.TensorDataset(X, a, f)
    data_iter = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_iter


# 丢弃h小于阈值的基函数
def drop_bf(net, tol1=0.00001):
    net = net.cpu()
    print(f'-----丢弃前基函数个数：{net.hight.shape[0]}')
    # show_u(net.center,net.hight,net.width)
    c, h, w = net.center.detach(), net.hight.detach(), net.width.detach()
    index = torch.where(abs(h) > tol1)[0]
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


# 计算边界损失
def get_bound_loss(c, h, w, dx=0.001,batch_size=512):  # c:n*2,,,h:n,,,w:n
    Y = torch.linspace(0, 1, int(1 / dx) + 1).to(device)
    X = torch.zeros_like(Y).to(device)
    dataset = Data.TensorDataset(X.view(-1, 1),Y.view(-1, 1))
    data_iter = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    for x,y in data_iter:
        X_1 = torch.cat((x.view(-1, 1), y.view(-1, 1)), dim=1)  # (0,y)
        X_2 = torch.cat(((x + 1).view(-1, 1), y.view(-1, 1)), dim=1)  # (1,y)
        X_3 = torch.cat((y.view(-1, 1), x.view(-1, 1)), dim=1)  # (x,0)
        X_4 = torch.cat((y.view(-1, 1), (x + 1).view(-1, 1)), dim=1)  # (x,1)
        bound_loss = ((get_u(X_1, c, h, w) - 1) ** 2).mean() + ((get_u(X_2, c, h, w) - 1) ** 2).mean() + \
                     ((get_u(X_3, c, h, w) - 1) ** 2).mean() + ((get_u(X_4, c, h, w) - 1) ** 2).mean()
        break
    return bound_loss





def show_err():
    data = np.load('output/exam5/err_rec_%.3f.npz' % eps)
    err_H1 = data['H1']
    err_L2 = data['L2']
    err_Li = data['L_inf']
    epoch_rec = data['epoch_rec']
    epoch_sparse = data['epoch_sparse']
    N_rec = data['N']

    plt.grid(linestyle='--')
    plt.plot(epoch_rec, err_H1)
    plt.plot(epoch_rec, err_L2)
    plt.plot(epoch_rec, err_Li)
    plt.semilogy()
    plt.legend(['$H^1$', '$L^2$', r'$L^\infty$'])
    plt.xlabel('Iter', fontsize=16)
    plt.ylabel('Relative error', fontsize=16)
    #plt.savefig(r'D:\cc\Desktop\tex\SRBF\imgs\1d\exam3\err_draw_0.002.png')
    plt.show()
    plt.grid(linestyle='--')
    plt.plot(epoch_rec, N_rec)
    plt.xlabel('Iter', fontsize=16)
    plt.ylabel('The number of RBF', fontsize=16)
    #plt.savefig(r'D:\cc\Desktop\tex\SRBF\imgs\1d\exam3\N_draw_0.002.png')
    plt.show()



def show_u(net, dx=0.001):
    x0 = torch.linspace(0, 1, int(1 / dx + 1))
    x = x0.reshape(-1,1)
    val = 0.5
    y = val*torch.ones_like(x)
    # print(x,np.unique(x),y,np.unique(y))
    u_FDM = np.load("2d_FDM/2d_exam5_FDM_%.3f.npy" % eps)[::2,::2]
    ux_FDM = np.load("2d_FDM/2d_exam5_FDM_x_%.3f.npy" % eps)[::2,::2]
    uy_FDM = np.load("2d_FDM/2d_exam5_FDM_y_%.3f.npy" % eps)[::2,::2]
    X = torch.cat((x, y), dim=1) 
    with torch.no_grad():
        u = get_u(X, net.center, net.hight, net.width).flatten()
        ux = get_ux(X, net.center, net.hight, net.width).flatten()
        uy = get_uy(X, net.center, net.hight, net.width).flatten()
    plt.plot(x0,u)
    plt.plot(x0,u_FDM[:,int(1000*val)],'r')
    plt.legend(['SRBF','FDM'])
    plt.show()
    plt.plot(x0[1:-1],ux[1:-1])
    plt.plot(x0[1:-1],ux_FDM[1:-1,int(1000*val)],'r')
    plt.legend(['SRBF','FDM'])
    plt.show()
    plt.plot(x0[1:-1],uy[1:-1])
    plt.plot(x0[1:-1],uy_FDM[1:-1,int(1000*val)],'r')
    plt.legend(['SRBF','FDM'])
    plt.show()




# 定义模型
class SRBF2d(nn.Module):
    def __init__(self, N1):
        super(SRBF2d, self).__init__()
        self.hight = nn.Parameter(torch.rand(N1))  # 高斯分布0,1
        self.center = nn.Parameter(torch.rand(N1, 2))  # [0,1]均匀分布
        self.width = nn.Parameter(5 * torch.rand(N1, 2) / eps)

        self.hight2 = nn.Parameter(torch.rand(N1))  # 高斯分布0,1
        self.center2 = nn.Parameter(torch.rand(N1, 2))  # [0,1]均匀分布
        self.width2 = nn.Parameter(5 * torch.rand(N1, 2) / eps)

        self.hight3 = nn.Parameter(torch.rand(N1))  # 高斯分布0,1
        self.center3 = nn.Parameter(torch.rand(N1, 2))  # [0,1]均匀分布
        self.width3 = nn.Parameter(5 * torch.rand(N1, 2) / eps)

    def forward(self, x):
 
        ux = get_ux(x, self.center, self.hight, self.width)
        uy = get_uy(x, self.center, self.hight, self.width)

        P = get_u(x, self.center2, self.hight2, self.width2)
        Px = get_ux(x, self.center2, self.hight2, self.width2)


        Q = get_u(x, self.center3, self.hight3, self.width3)
        Qy = get_uy(x, self.center3, self.hight3, self.width3)


        return ux, uy, P, Px, Q, Qy


def get_err(net):
    dx = 0.001
    net = net.to(device)
    u_FDM = np.load("2d_FDM/2d_exam5_FDM_%.3f.npy" % eps)[::2,::2]
    ux_FDM = np.load("2d_FDM/2d_exam5_FDM_x_%.3f.npy" %eps)[::2,::2][1:-1,1:-1].flatten()
    uy_FDM = np.load("2d_FDM/2d_exam5_FDM_y_%.3f.npy" %eps)[::2,::2][1:-1,1:-1].flatten()
    u_FDM = u_FDM.flatten()
    x0 = torch.linspace(0, 1, int(1 / dx + 1))
    x,y = torch.meshgrid(x0,x0)
    x,y = x.flatten(),y.flatten()
    X = torch.cat((x.view(-1, 1), y.view(-1, 1)), dim=1)  # [[0,0],[0,0.001],...]
    c = 2 * math.pi / eps
    print(1)
    X_set = Data.TensorDataset(X)
    X_dataloader = Data.DataLoader(dataset=X_set, batch_size=5000, shuffle=False)
    u,Pa,Qa = [],[],[]
    for X_part in X_dataloader:
        X_part[0] = X_part[0].to(device)
        u_part = get_u(X_part[0], net.center, net.hight, net.width).cpu().detach().numpy()
        Pa_part = get_u(X_part[0], net.center2, net.hight2, net.width2).cpu().detach().numpy()
        Qa_part = get_u(X_part[0], net.center3, net.hight3, net.width3).cpu().detach().numpy()
        a = 2 + torch.sin(2 * math.pi * (X_part[0][:,0] + X_part[0][:,1]) / eps)
        u.extend(u_part)
        Pa.extend(Pa_part/a.cpu().numpy())
        Qa.extend(Qa_part/a.cpu().numpy())
    print(2)
    ux = np.array(Pa).reshape(-1, int(1 / dx + 1))
    uy = np.array(Qa).reshape(-1, int(1 / dx + 1))
    u = np.array(u)
    L2_err = np.sqrt(((u - u_FDM) ** 2).sum()) / np.sqrt((u_FDM ** 2).sum())
    H1_err = np.sqrt(((((ux[1:-1, 1:-1]).flatten() - ux_FDM) ** 2).sum() + (
            ((uy[1:-1, 1:-1]).flatten() - uy_FDM) ** 2).sum() + ((u - u_FDM) ** 2).sum())) / (
                 np.sqrt((ux_FDM ** 2).sum() + (
                         uy_FDM ** 2).sum() + (u_FDM ** 2).sum()))
    L_inf_err = np.max(abs(u - u_FDM)) / np.max(abs(u_FDM))
    return L2_err, H1_err, L_inf_err




# 定义训练函数
def train(net, epochs, data_iter, lr, eps):
    print('Training on %s' % device)
    net = net.to(device=device)
    optimizer = optims.Adam(net.parameters(),lr=lr)
    loss_rec = [0.0]
    thres, flag, l_sums,lam_r = 0.0, True, 0.0,0.001
    err_L2_rec,err_H1_rec,err_Li_rec,epoch_rec,N_rec = [],[],[],[],[]
    epoch_sparse,epoch_flag = 0,True
    t_all = 0.0
    epoch_end = epochs
    L2, H1, L_inf = get_err(net)
    print('L2:', L2, 'H1:', H1, 'L_inf:', L_inf)
    err_L2_rec.append(L2)
    err_H1_rec.append(H1)
    err_Li_rec.append(L_inf)
    epoch_rec.append(0)
    N_rec.append(net.hight.detach().shape[0])
    np.savez('output/exam5/err_rec_%.3f.npz'%eps,H1=err_H1_rec,L2=err_L2_rec,L_inf=err_Li_rec,epoch_rec=epoch_rec,epoch_sparse=epoch_sparse,N=N_rec)
    net = net.to(device)
    for epoch in range(1, epochs + 1):
        l_P_sum = 0.0
        l_Q_sum = 0.0
        l_f_sum = 0.0
        l_bound_sum = 0.0
        t1 = time.time()
        for x, a, f in data_iter:
            x = x.to(device=device)
            a = a.to(device=device)
            f = f.to(device=device)
            ux, uy, P, Px, Q, Qy = net(x)
            l_P = ((ux - P / a) ** 2).mean()
            l_Q = ((uy - Q / a) ** 2).mean()
            l_f = ((Px + Qy - f) ** 2).mean()
            l_bound = get_bound_loss(net.center, net.hight, net.width)
            l = (l_P + l_Q) + 0.004*l_f + 20*l_bound    #0.1,0.1,0.1,0.06,0.06,0.006
            if l_sums<thres:
                print('----加上正则项-----')
                l = l+ lam_r*net.hight.norm(1)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_P_sum += l_P.cpu().item()
            l_Q_sum += l_Q.cpu().item()
            l_f_sum += l_f.cpu().item()
            l_bound_sum += l_bound.cpu().item()
        t2 = time.time()
        t_all += t2-t1
        l_sums = l_P_sum + l_Q_sum + l_f_sum + l_bound_sum
        if (epoch%40==0):
            if optimizer.param_groups[0]['lr']>0.0001:
                optimizer.param_groups[0]['lr'] = 0.1*optimizer.param_groups[0]['lr']
        print('thres=%.3f,t_all:%.3f,t_epo:%.3f, eps=%.3f,epoch %d,l(P):%f,l(Q):%f,,l(f):%f,l(bound):%f,l(all):%f'
              % (thres,t_all,t2-t1, eps, epoch, l_P_sum, l_Q_sum, l_f_sum, l_bound_sum, l_sums))

        if epoch == epochs-50:
            thres = 0.0
        if epoch%5==0:
            L2, H1, L_inf = get_err(net)
            print('L2:', L2, 'H1:', H1, 'L_inf:', L_inf)
            err_L2_rec.append(L2)
            err_H1_rec.append(H1)
            err_Li_rec.append(L_inf)
            epoch_rec.append(epoch)
            N_rec.append(net.hight.detach().shape[0])
            np.savez('output/exam5/err_rec_%.3f.npz'%eps,H1=err_H1_rec,L2=err_L2_rec,L_inf=err_Li_rec,epoch_rec=epoch_rec,epoch_sparse=epoch_sparse,N=N_rec)
            net = net.to(device)
        if (epoch % 10 == 0):
            loss_rec.append(l_sums)
            if thres>0.0:
                net = drop_bf(net)
                lr_new = optimizer.param_groups[0]['lr']
                optimizer = optims.Adam(net.parameters(),lr=lr_new)
            torch.save(net.state_dict(), 'model/exam5/2d_exam5_draw_%.3f.pth' % (eps))
            net = net.cpu()
            if (abs(loss_rec[-1]-loss_rec[-2])<0.1)&flag:
                thres = l_sums+0.1
                flag = False
            index = torch.where(abs(net.hight.data) > 0.00001)[0]
            print('-----基函数个数：{}-----'.format(index.shape[0]),'net.hight.shape:%d'%net.hight.shape[0])
            print('--------lr:{0:.6f}'.format(optimizer.param_groups[0]['lr']))
            print(loss_rec)
            net = net.to(device)


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
    # print(net.width)
    return net





if __name__ == "__main__":
    batch_size = 1024
    n = 30000   #1000,1000,2000,5000,15000,30000
    epochs = 300
    eps = 0.01
    lr, dx = 0.1, 0.002  # min(0.1*eps,0.002)

    data_iter = dataloader(dx, batch_size, eps)
    net = SRBF2d(n)
    if os.path.exists('model/exam5/2d_exam5_draw_%.3f_1.pth' % (eps)):
        print('加载已训练网络,eps:%.3f,n:%d' % (eps, n))
        ckpt = torch.load('model/exam5/2d_exam5_draw_%.3f_1.pth' % (eps), map_location='cpu')
        net = load_ckpt(net, ckpt)
    print('-----基函数个数：{}-----'.format(net.hight.detach().shape[0]))
    #train(net, epochs, data_iter, lr, eps)
    h = net.hight.data
    index = torch.where(abs(h) > 0.00001)[0]
    print(index.shape)
    net = net.cpu()
    #L2, H1, L_inf = get_err(net)
    #print('L2:', L2, 'H1:', H1, 'L_inf:', L_inf)
    show_err()
    #show_u(net)



