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
import torch.nn.functional as tnf
torch.manual_seed(1)
torch.set_printoptions(precision=8)


# generate training data
def dataloader(num_points, batch_size, eps):
    x = torch.rand(num_points,1)
    x.requires_grad_(True)
    b = torch.fmod(x,torch.tensor(eps))
    a = 1*(torch.ge(b,0)&torch.lt(b,0.5*eps))+10*(torch.ge(b,0.5*eps)&torch.lt(b,eps))
    f = torch.ones_like(x)
    dataset = Data.TensorDataset(x, a,f)
    data_iter = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_iter


#load the FDM solution
def generate_FDM(eps):
    if os.path.exists('1d_FDM/1d_exam4_%.3f.npy'% (eps)): 
        u_FDM = np.load('1d_FDM/1d_exam4_%.3f.npy' % (eps)).flatten()
        ux_FDM = np.zeros_like(u_FDM)
        ux_FDM[1:-1] = (u_FDM[2:] - u_FDM[:-2]) / 0.0002
        ux_FDM[0] = -1.5 * u_FDM[0] / 0.0001 + 2 * u_FDM[1] / 0.0001 - 0.5 * u_FDM[2] / 0.0001
        ux_FDM[-1] = 1.5 * u_FDM[-1] / 0.0001 - 2 * u_FDM[-2] / 0.0001 + 0.5 * u_FDM[-3] / 0.0001
    else:
        raise ValueError("no the FDM solution")
    return u_FDM, ux_FDM


#calculate the boundary loss
def get_bd_loss(net,scale,device):
    xa = torch.zeros(1).to(device)
    xb = torch.ones(1).to(device)
    la = (net(xa,scale)-1)**2
    lb = (net(xb,scale)-1)**2
    return la+ lb

#use autograd to calculate the gradients
def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True,only_inputs=True)[0]                          
    else:
        return gradients(gradients(u, x), x, order=order - 1)


#calculate the three relative errors
def get_err(net,scale,eps):
    net = net.cpu()
    x = torch.linspace(0, 1, 10001).view(-1,1)
    x.requires_grad_(True)
    u = net(x,scale)
    ux = gradients(u,x,order=1)
    u,ux = u.detach().numpy().flatten(),ux.detach().numpy().flatten()
    u_FDM,ux_FDM = generate_FDM(eps)
    L2_err = np.sqrt(((u - u_FDM) ** 2).sum()) / np.sqrt((u_FDM ** 2).sum())
    H1_err = np.sqrt((((ux - ux_FDM) ** 2).sum() + ((u - u_FDM) ** 2).sum())) / (
        np.sqrt((ux_FDM ** 2).sum() + (u_FDM ** 2).sum()))
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
    def __init__(self, indim=1, outdim=1, hidden_units=None, actName2in='s2relu', actName='s2relu',
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
	


#the training process of MscaleDNN
def train(net, epochs, data_iter,scale,device, lr, eps,lam_b):
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
            u = net(x,scale)
            ux = gradients(u,x,order=1)
            l_r = torch.mean(0.5*ux*a*ux-f*u)
            l_bd = get_bd_loss(net,scale,device)
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
            #torch.save(net.state_dict(), 'model/1d_exam4_%.3f.pth' % (eps))
            L2, H1, L_inf = get_err(net,scale,eps)
            print('Rel. L2:', L2,'Rel. L_inf:', L_inf, 'Rel. H1:', H1 )
            print('--------lr:{0:.6f}'.format(optimizer.param_groups[0]['lr']))
            net = net.to(device)

#show the solution 
def show(net,scale,eps):
    net = net.cpu()
    x =torch.linspace(0,1,10001).view(-1,1)
    x.requires_grad_(True)
    u = net(x,scale).flatten()
    ux = gradients(u,x,order=1)
    x,u,ux = x.detach().numpy(),u.detach().numpy(),ux.detach().numpy()
    u_FDM,ux_FDM = generate_FDM(eps)
    plt.plot(x,u)
    plt.plot(x,u_FDM)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend(['MscaleDNN','FDM'])
    plt.show()
    plt.plot(x,ux,'r')
    plt.plot(x,ux_FDM)
    plt.xlabel('x')
    plt.ylabel(r'$u_x$')
    plt.legend(['MscaleDNN','FDM'])
    plt.show()

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    eps = args.eps
    num_points = args.num_points
    data_iter = dataloader(num_points,batch_size,eps)
    hidden_units = args.hidden_units
    scale = np.arange(int(1/eps))
    scale[0] += 1.0
    lam_b = args.lam_b
    net = MscaleDNN(indim=1, outdim=1, hidden_units=hidden_units, actName2in='s2relu', actName='s2relu',
                 actName2out='linear') 
    if args.pretrained:
        if os.path.exists('model/1d_exam4_%.3f.pth' % (eps)):
            net.load_state_dict(torch.load('model/1d_exam4_%.3f.pth' % (eps)))
            print('load the trained model')
        else:
            print('no trained model')
    train(net, epochs, data_iter,scale,device, lr, eps,lam_b)
    net = net.cpu()
    show(net,scale,eps)
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameters for MscaleDNN: %d" % (total))
    L2, H1, L_inf = get_err(net,scale,eps)
    print('Rel. L2:', L2,'Rel. L_inf:', L_inf, 'Rel. H1:', H1 )


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--eps', type=float, default=0.5) #[0.5,0.1,0.05,0.01,0.005,0.002]
    parser.add_argument('--hidden_units', type=tuple, default=(1000, 200, 150, 150,100,50, 50))
    parser.add_argument('--batch_size',type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_points', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lam_b', type=float, default=200.0)
    parser.add_argument('--pretrained', type=str, default=False)
    args = parser.parse_args()
    main(args)

"""    m    params       L2              H1            L_inf
0.5:  16     1137      L2: 0.016678978196065165 H1: 0.09213136744971263 L_inf: 0.026931054070557382
0.1:   16     1137     L2: 0.03060155469842837 H1: 0.15829160233656406 L_inf: 0.04129516423688839
0.05:  32     4321     L2: 0.030923513739831616 H1: 0.1596840707567103 L_inf: 0.04172218947899356
0.01:  32     4321     L2: 0.03107623670961607 H1: 0.15952726207040688 L_inf: 0.041883226554335976
0.005:  64   16833      L2: 0.031044448726389985 H1: 0.1585458486949831 L_inf: 0.04192295224343453
0.002:  64    16833     L2: 0.031068505110652976 H1: 0.1557872051619962 L_inf: 0.04195359130002731
"""
