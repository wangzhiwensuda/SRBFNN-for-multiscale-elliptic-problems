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
import torch.nn.functional as tnf
torch.manual_seed(1)
torch.set_printoptions(precision=8)


# generate training data
def dataloader(h, batch_size, eps):
    x0 = torch.linspace(0, 1, int(1 / h + 1))
    x,y = torch.meshgrid(x0,x0)
    x,y = x.flatten(),y.flatten()
    x,y = x.view(-1,1),y.view(-1,1)
    X = torch.cat((x,y),dim=1)
    dataset = Data.TensorDataset(X)
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
def get_bd_loss(net,scale,device,batch_size_bd=512):
    x = torch.linspace(0,1,batch_size_bd).view(-1,1).to(device)
    y = torch.zeros_like(x).to(device)
    X_up = torch.cat([x,y+1],dim=1) #(x,1)
    X_down = torch.cat([x,y],dim=1) #(x,0)
    X_left = torch.cat([y,x],dim=1) ##(0,y)
    X_right = torch.cat([y+1,x],dim=1) #(1,y)
    l_up = ((net(X_up,scale)-0)**2).mean()
    l_down = ((net(X_down,scale)-0)**2).mean()
    l_left = ((net(X_left,scale)-0)**2).mean()
    l_right = ((net(X_right,scale)-0)**2).mean()
    return l_up+l_down+l_left+l_right

#use autograd to calculate the gradients
def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),create_graph=True,only_inputs=True)[0]                         
    else:
        return gradients(gradients(u, x), x, order=order - 1)


#calculate the three relative errors
def get_err(net,scale,eps):
    net = net.cpu()
    x0 = torch.linspace(0, 1, 1001)
    x,y = torch.meshgrid(x0,x0)
    x,y = x.reshape(-1,1),y.reshape(-1,1)
    X = torch.cat([x,y],dim=1)
    dataset = Data.TensorDataset(X)
    data_iter = Data.DataLoader(dataset=dataset, batch_size=5000, shuffle=False)
    u,ux,uy = [],[],[]
    for X_part in data_iter:
        X_p = X_part[0]
        X_p.requires_grad_(True)
        u_part = net(X_p,scale)
        u_grad_part = gradients(u_part,X_p,order=1)
        u.extend(u_part.detach().numpy())
        ux.extend(u_grad_part[:,0].detach().numpy())
        uy.extend(u_grad_part[:,1].detach().numpy())
    u,ux,uy = np.array(u),np.array(ux).reshape(-1,1001)[1:-1,1:-1],np.array(uy).reshape(-1,1001)[1:-1,1:-1]
    u,ux,uy = u.flatten(),ux.flatten(),uy.flatten()
    u_FDM,ux_FDM,uy_FDM = generate_FDM(eps)
    u_FDM,ux_FDM,uy_FDM = u_FDM.flatten(),ux_FDM.flatten(),uy_FDM.flatten()
    L2_err = np.sqrt(((u - u_FDM) ** 2).sum()) / np.sqrt((u_FDM ** 2).sum())
    H1_err = np.sqrt(((u - u_FDM) ** 2).sum()+((ux - ux_FDM) ** 2).sum() + ((uy - uy_FDM) ** 2).sum()
             ) / np.sqrt((u_FDM ** 2).sum()+(ux_FDM ** 2).sum() + (uy_FDM ** 2).sum())                 
    L_inf_err = np.max(abs(u - u_FDM)) / np.max(abs(u_FDM))
    return L2_err, H1_err, L_inf_err



#Activation functions
class my_actFunc(nn.Module):
    def __init__(self, actName='linear'):
        super(my_actFunc, self).__init__()
        self.actName = actName
    def forward(self, x_input):
        if str.lower(self.actName) == 'tanh':
            out_x = torch.tanh(x_input)
        elif str.lower(self.actName) == 'srelu':
            out_x = tnf.relu(x_input)*tnf.relu(1-x_input)
        elif str.lower(self.actName) == 's2relu':
            out_x = tnf.relu(x_input)*tnf.relu(1-x_input)*torch.sin(2*np.pi*x_input)
        else:
            out_x = x_input
        return out_x

#the model of MscaleDNN
class MscaleDNN(nn.Module):
    def __init__(self, indim=2, outdim=1, hidden_units=None, actName2in='s2relu', actName='s2relu',
                 actName2out='linear'):       
        super(MscaleDNN, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.dense_layers = nn.ModuleList()

        input_layer = nn.Linear(in_features=indim, out_features=hidden_units[0])
        nn.init.xavier_normal_(input_layer.weight)
        nn.init.uniform_(input_layer.bias, -1, 1)
        self.dense_layers.append(input_layer)

        for i_layer in range(len(hidden_units)-1):
            hidden_layer = nn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer+1])
            nn.init.xavier_normal_(hidden_layer.weight)
            nn.init.uniform_(hidden_layer.bias, -1, 1)
            self.dense_layers.append(hidden_layer)

        out_layer = nn.Linear(in_features=hidden_units[-1], out_features=outdim)
        nn.init.xavier_normal_(out_layer.weight)
        nn.init.uniform_(out_layer.bias, -1, 1)
        self.dense_layers.append(out_layer)

    def forward(self, inputs, scale=None):  
        # ------ dealing with the input data ---------------
        dense_in = self.dense_layers[0]
        H = dense_in(inputs)  
        Unit_num = int(self.hidden_units[0] / len(scale)) 
        mixcoe = np.repeat(scale, Unit_num) 


        mixcoe = mixcoe.astype(np.float32)
        torch_mixcoe = torch.from_numpy(mixcoe)
        if inputs.is_cuda:
            torch_mixcoe = torch_mixcoe.cuda()
        #print('-----',H,torch_mixcoe)
        H = self.actFunc_in(H*torch_mixcoe)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units)-1):
            H_pre = H
            dense_layer = self.dense_layers[i_layer+1]
            H = dense_layer(H)
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        dense_out = self.dense_layers[-1]
        H = dense_out(H)
        H = self.actFunc_out(H)
        return H



def train(net, epochs, data_iter,scale,batch_size_bd,device, lr, eps,lam_b):
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
        for X in data_iter:
            X = X[0].to(device)
            X.requires_grad_(True)
            a = (1.5 + torch.sin(c1 * X[:,0])) / (1.5 + torch.sin(c1 * X[:,1])) + \
                (1.5 + torch.sin(c1 * X[:,1])) / (1.5 + torch.cos(c1 * X[:,0])) + torch.sin(4 * X[:,0] ** 2 * X[:,1] ** 2) + 1
            a = a.view(-1,1)
            f = 10 * torch.ones_like(X[:,0]).view(-1,1)
            u = net(X,scale)
            u_grad= gradients(u,X,order=1)
            temp = torch.sum(0.5*u_grad*a*u_grad,dim=1)+f*u
            l_r = torch.mean(temp)
            l_bd = get_bd_loss(net,scale,device,batch_size_bd)
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
            L2, H1, L_inf = get_err(net,scale,eps)
            print('Rel. L2:', L2,'Rel. L_inf:', L_inf, 'Rel. H1:', H1 )
            print('--------lr:{0:.6f}'.format(optimizer.param_groups[0]['lr']))
            net = net.to(device)

# show point-wise error |u^S-u^F|
def show_pointwise_error(net,scale,eps,device, dx=0.001):
    x0 = torch.linspace(0, 1, int(1 / dx + 1))
    x, y = torch.meshgrid(x0, x0)
    x, y = x.flatten(), y.flatten()
    X = torch.cat((x.view(-1, 1), y.view(-1, 1)), dim=1)  # [[0,0],[0,0.001],...]
    with torch.no_grad():
        u = net(X,scale)
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
    hidden_units = args.hidden_units
    scale = np.arange(int(1/eps))
    scale[0] += 1.0
    lam_b = args.lam_b
    net = MscaleDNN(indim=2, outdim=1, hidden_units=(1000, 400, 300, 300,200,100, 100), actName2in='s2relu', actName='s2relu',
                 actName2out='linear')
    if args.pretrained:
        if os.path.exists('model/2d_exam6_%.3f.pth' % (eps)):
            net.load_state_dict(torch.load('model/2d_exam6_%.3f.pth' % (eps)))
            print('load the trained model')
        else:
            print('no trained model')
    train(net, epochs, data_iter,scale,batch_size_bd,device, lr, eps,lam_b)
    net = net.cpu()
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameters for MscaleDNN: %d" % (total))
    L2, H1, L_inf = get_err(net,scale,eps)
    print('Rel. L2:', L2,'Rel. L_inf:', L_inf, 'Rel. H1:', H1 )
    show_pointwise_error(net,scale,eps,device)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--eps', type=float, default=0.5) #[0.5,0.2,0.1,0.05,0.02,0.01]
    parser.add_argument('--hidden_units', type=tuple, default=(1000,400,300,300,200,100,100)) 
    parser.add_argument('--batch_size',type=int, default=2048)
    parser.add_argument('--batch_size_bd',type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--h', type=int, default=0.002)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lam_b', type=float, default=200.0)
    parser.add_argument('--pretrained', type=str, default=False)
    args = parser.parse_args()
    main(args)

"""
L2: 0.007884654027324337 H1: 0.05978841867577853 L_inf: 0.01295674536277825
L2: 0.00846719222974051 H1: 0.0992253637637821 L_inf: 0.019566449576509114
L2: 0.008229202999159383 H1: 0.11155250241598048 L_inf: 0.014419879966787577
L2: 0.030960306390050832 H1: 0.21635653770659233 L_inf: 0.03645468315853878
L2: 0.020661015329247325 H1: 0.20194844901742054 L_inf: 0.027902546402035378
L2: 0.02589434797125868 H1: 0.20776159471279163 L_inf: 0.0314912066822129
"""
