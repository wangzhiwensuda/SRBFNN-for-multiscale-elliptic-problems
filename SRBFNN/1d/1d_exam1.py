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
torch.set_printoptions(precision=8)

#判断是否有GPU，如果有则选用编号为'0'的GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# 生成训练数据，并加载FDM参考解
def dataloader(h, batch_size, eps):
    x = torch.linspace(0, 1, int(1 / h + 1))
    a = 2 + torch.sin(2 * math.pi * x / eps)
    dataset = Data.TensorDataset(x, a)
    data_iter = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    u_FDM = np.load('1d_FDM/1d_exam1_%.3f.npy' % (eps)).flatten()
    ux_FDM = np.zeros_like(u_FDM)
    ux_FDM[1:-1] = (u_FDM[2:] - u_FDM[:-2]) / (2 * h)
    ux_FDM[0] = -1.5 * u_FDM[0] / h + 2 * u_FDM[1] / h - 0.5 * u_FDM[2] / h
    ux_FDM[-1] = 1.5 * u_FDM[-1] / h - 2 * u_FDM[-2] / h + 0.5 * u_FDM[-3] / h
    return data_iter, u_FDM, ux_FDM


#删除权重(用hight表示)小于tol2的RBF
def drop_bf(net,tol2=0.0001):
    net = net.cpu()
    print(f'-----丢弃前基函数个数：{net.hight.shape[0]}')
    c, h, w = net.center.detach(), net.hight.detach(), net.width.detach()
    index = torch.where(abs(h) > tol2)[0]
    c1 = torch.index_select(c, 0, index)
    h1 = torch.index_select(h, 0, index)
    w1 = torch.index_select(w, 0, index)
    net.center = nn.Parameter(c1)
    net.hight = nn.Parameter(h1)
    net.width = nn.Parameter(w1)
    print(f'丢弃后基函数个数------：{net.hight.shape[0]}')
    return net.to(device)



# RBFNN，x表示输入，c,h,w分别表示基函数的中心，权重和宽度
def get_u(x, c, h, w):  
    c1 = (x.view(-1, 1) - c.view(-1,1,1)) ** 2 
    d2 = (w** 2).view(-1,1,1)   
    r = -torch.matmul(c1,d2)
    m = torch.exp(r) 
    output = torch.matmul(h,m.squeeze(-1))
    return output.flatten()

# RBFNN对x的偏导数
def get_ux(x, c, h, w):
    c1 = (x.view(-1, 1) - c.view(-1,1,1)) 
    d2 = (w** 2).view(-1,1,1) 
    r = -torch.matmul(c1** 2, d2)
    m = -2*torch.matmul(c1,d2)*torch.exp(r)  
    output = torch.matmul(h,m.squeeze(-1))
    return output.flatten()


#计算三个相对误差
def get_err(net):
    dx = 0.0001
    net = net.cpu()
    x = torch.linspace(0, 1, int(1 / dx + 1))
    ax = 2 + torch.sin(2 * math.pi * x / eps)
    c, h, w = net.center, net.hight, net.width
    u_SRBF = get_u(x, c, h, w).detach().numpy()
    p = get_u(x, net.center2, net.hight2, net.width2).detach().numpy()
    pa = p/ax
    u,ux = u_SRBF.flatten(),pa.flatten()
    L2_err = np.sqrt(((u - u_FDM) ** 2).sum())/np.sqrt((u_FDM**2).sum())
    H1_err = np.sqrt((((ux - ux_FDM) ** 2).sum()  + ((u - u_FDM) ** 2).sum()))/(np.sqrt((ux_FDM** 2).sum()+ (u_FDM ** 2).sum()))
    L_inf_err = np.max(abs(u - u_FDM))/np.max(abs(u_FDM))
    return L2_err, H1_err,L_inf_err

#计算边界损失
def get_bd_loss(net):
    x0 = torch.zeros(1).to(device=device)
    l_a = get_u(x0, net.center, net.hight, net.width) - 1
    l_b = get_u(x0 + 1, net.center, net.hight, net.width) - 1
    l_bd = (l_b ** 2 + l_a ** 2)
    return l_bd

# 定义模型
class SRBF(nn.Module):
    def __init__(self, n, eps):
        super(SRBF, self).__init__()
        torch.manual_seed(1)
        self.hight = nn.Parameter(torch.rand(n))
        self.center = nn.Parameter(torch.rand(n, 1))
        self.width = nn.Parameter(torch.rand(n, 1) * (5 / eps))

        self.hight2 = nn.Parameter(torch.rand(n))
        self.center2 = nn.Parameter(torch.rand(n, 1))
        self.width2 = nn.Parameter(torch.rand(n, 1) * (5 / eps))


    def forward(self, x):
        ux = get_ux(x, self.center, self.hight, self.width)
        v = get_u(x, self.center2, self.hight2, self.width2)
        vx = get_ux(x, self.center2, self.hight2, self.width2)
        return v, vx, ux


def show_err():
    eps_all = [0.5,0.1,0.05,0.01,0.005,0.002]
    x = np.linspace(0, 1, 10001)
    plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示符号
    plt.grid(linestyle="--")
    for eps in eps_all:
        net = SRBF(1, eps)
        net = load_pth(net,eps)
        y_SRBF = get_u(torch.Tensor(x), net.center, net.hight, net.width).detach().numpy()
        u_FDM = np.load(r'1d_FDM\1d_q1_%.3f.npy' % (eps)).flatten()
        plt.plot(x,abs(y_SRBF-u_FDM),lw=0.6)
    plt.xlabel('x',fontsize=16)
    plt.ylabel('error',fontsize=16)
    plt.legend([r'$\varepsilon$=0.5',r'$\varepsilon$=0.1',r'$\varepsilon$=0.05',r'$\varepsilon$=0.01',r'$\varepsilon$=0.005',r'$\varepsilon$=0.002'])
    #plt.savefig(r'D:\cc\Desktop\tex\SRBF\imgs\1d\exam1\err.png',dpi=200)
    plt.show()




# 定义训练函数
def train(net, epochs, data_iter, lr, eps):
    print('Training on %s' % device)
    optimizer = optims.Adam(net.parameters(), lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.1) #eps小的时候,step_size可以调小一点
    t_all = 0.0
    net = net.to(device=device)
    loss_rec = [0.0]
    l_sums =0.0
    flag, thres,lam_r = True,0.0,0.1  #eps越小,正则项系数lam_r可以设置越小
    for epoch in range(1, epochs + 1):
        l_v_sum = 0.0
        l_vx_sum = 0.0
        l_bd_sum = 0.0
        t1 = time.time()
        for x, ax in data_iter:
            x = x.to(device=device)
            ax = ax.to(device=device)
            v, vx, ux = net(x)
            l_bd = get_bd_loss(net)
            l_vx = ((vx + 1) ** 2).mean()
            l_v = ((ux - v/ax) ** 2).mean()
            l = l_v + l_vx + 10*l_bd
            if (l_sums < thres):
                print('-----加上正则项,进行稀疏优化------')
                l = l + lam_r * (net.hight.norm(1)+net.hight2.norm(1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_v_sum += l_v.cpu().item()
            l_vx_sum += l_vx.cpu().item()
            l_bd_sum += l_bd.cpu().item()
        t2 = time.time()
        t_all += t2-t1
        l_sums = l_v_sum + l_vx_sum+l_bd_sum
        if (optimizer.param_groups[0]['lr']>0.0001):
            scheduler.step()
        print('eps=%.3f,t_all=%.3f,t_epoch=%.3f,th=%.4f,epoch %d,l_v:%f,l_vx:%f,l_bd:%f,l_all:%f' 
            % (eps,t_all,t2-t1, thres, epoch, l_v_sum, l_vx_sum,l_bd_sum, l_sums))
        if epoch == (epochs-2000): #最后两千epoch不加正则项优化
            thres = 0.0
        if (epoch % 100 == 0):
            loss_rec.append(l_sums)
            if thres>0.0:
                net = drop_bf(net)
                lr_new = optimizer.param_groups[0]['lr']
                optimizer = optims.Adam(net.parameters(),lr=lr_new)
            torch.save(net.state_dict(), 'pth_1d/Adam/exam1/1d_exam1_%.3f.pth' % (eps))
            net = net.cpu()
            L2,H1,L_inf = get_err(net)
            print('L2:', L2, 'H1:', H1, 'L_inf:', L_inf)
            if (abs(loss_rec[-1] - loss_rec[-2]) < -0.01) & flag:
                thres = loss_rec[-1] + 0.04
                flag = False
            print('-----基函数个数：{}-----'.format(net.hight.detach().shape[0]))
            print('--------lr:{0:.6f}'.format(optimizer.param_groups[0]['lr']))
            net = net.to(device)


def load_pth(net,eps):
    ckpt = torch.load('pth_1d/Adam/exam1/1d_exam1_%.3f.pth' % (eps))
    net.center = nn.Parameter(ckpt['center'])
    net.hight = nn.Parameter(ckpt['hight'])
    net.width = nn.Parameter(ckpt['width'])
    net.center2 = nn.Parameter(ckpt['center2'])
    net.hight2 = nn.Parameter(ckpt['hight2'])
    net.width2 = nn.Parameter(ckpt['width2'])
    return net


if __name__ == "__main__":
    batch_size = 2048
    N = 1500 #100,200,300,500,1000,1500
    epochs = 5000
    lr, h = 0.00001, 0.0001
    eps = 0.002
    data_iter, u_FDM,ux_FDM = dataloader(h, batch_size, eps)
    net = SRBF(N,eps)
    if os.path.exists('pth_1d/Adam/exam1/1d_exam1_%.3f.pth' % (eps)):
        net = load_pth(net,eps)
        print('加载已训练网络,eps:%.3f,N:%d' % (eps, net.hight.shape[0]))
    print('-----基函数个数Ns：{}-----'.format(net.hight2.detach().shape[0]))
    print('-----基函数个数Ne：{}-----'.format(net.hight.detach().shape[0]))
    #train(net, epochs, data_iter, lr,eps)
    #show_err()
    # net = net.cpu()
    L2, H1, L_inf = get_err(net)
    print('L2:', L2, 'H1:', H1, 'L_inf:', L_inf)
    L2, H1, L_inf = get_err2(net)
    print('-----第二组,L2:', L2, 'H1:', H1, 'L_inf:', L_inf)



"""       基函数个数        相对L2误差              相对H1误差             相对L_inf误差
eps=0.5  100   17       L2: 3.100091946354364e-05 H1: 0.0013549734775101057 L_inf: 8.552424358788451e-05
eps=0.1  200   38       L2: 6.718142144073114e-05 H1: 0.004736413887358236 L_inf: 0.0002348345099692035
eps=0.05 300   66       L2: 8.311182670421178e-05 H1: 0.012750151960154273 L_inf: 0.0003291089057338414
eps=0.01  500  186       L2: 0.00010860751950087168 H1: 0.03529493572953952 L_inf: 0.0004052113196214109
eps=0.005 1000  367      L2: 7.48110200262596e-05 H1: 0.033856427189112095 L_inf: 0.0004106747668954035
eps=0.002  1500 750      L2: 0.0002541406072046563 H1: 0.04980268395146737 L_inf: 0.0003940354571117779
"""






