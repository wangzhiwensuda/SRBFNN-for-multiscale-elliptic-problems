import matplotlib.pyplot as plt
import numpy as np
import math

def exam1_fdm(h,eps):
    x = np.linspace(0,1,int(1/h+1))[1:-1]
    A = math.pi*np.cos(2*math.pi*x/eps)/(h*eps)
    B = (2+np.sin(2*math.pi*x/eps))/h**2
    a = A+B
    b = 2*B
    c = B-A
    m = x.shape[0]
    D = np.diag(-b,0)+np.diag(c[1:],-1)+np.diag(a[0:-1],1)
    f = -np.ones_like(x).reshape(-1,1)
    u1 = np.matmul(np.linalg.inv(D),f)
    u = np.ones(m+2)
    u[1:-1] = u1.flatten()+1
    # np.save('1d_FDM/1d_q1_%.3f.npy'%eps,u)

def exam2_fdm(h,eps):
    x = np.linspace(0, 1, int(1 / h + 1))[1:-1]
    A = (2*math.pi/eps)*np.cos(2*math.pi*x/eps)*np.cos(2*math.pi*x)-2*math.pi*np.sin(2*math.pi*x/eps)*np.sin(2*math.pi*x)
    B = 2+np.sin(2*math.pi*x/eps)*np.cos(2*math.pi*x)
    a = A / (2 * h) + B / h ** 2
    b = -2 * B / h ** 2
    c = B / h ** 2 - A / (2 * h)
    m = x.shape[0]
    D = np.diag(b, 0) + np.diag(c[1:], -1) + np.diag(a[0:-1], 1)
    f = -np.ones_like(x).reshape(-1, 1)
    u1 = np.matmul(np.linalg.inv(D), f)
    u = np.ones(m + 2)
    u[1:-1] = u1.flatten() + 1
    # np.save('1d_FDM/1d_q2_%.3f.npy' % eps, u)
    
def exam3_fdm(h,eps):
    x = np.linspace(0, 1, int(1 / h + 1))[1:-1]
    A = np.cos(2*math.pi*x+2*math.pi*x/eps)*(2*math.pi+2*math.pi/eps)
    B = 2+np.sin(2*math.pi*x/eps+2*math.pi*x)
    a = A / (2 * h) + B / h **2
    b = -2 * B / h ** 2
    c = B / h ** 2 - A / (2 * h)
    m = x.shape[0]
    D = np.diag(b, 0) + np.diag(c[1:], -1) + np.diag(a[0:-1], 1)
    f = -np.ones_like(x).reshape(-1, 1)
    u1 = np.matmul(np.linalg.inv(D), f)
    u = np.ones(m + 2)
    u[1:-1] = u1.flatten() + 1
    # np.save('1d_FDM/1d_q3_%.3f.npy' % eps, u)
    
    
def exam4_fdm(h,eps):
    x = np.linspace(0, 1, int(1 / h + 1))[1:-1]
    xc = np.arange(0.5 * h,1,h)
    xx = np.mod(xc, eps)
    A = 1.0 *(np.greater(xx,0) & np.less(xx, 0.5 * eps)) + 10.0* (np.greater(xx,0.5 * eps)  & np.less(xx, eps))
    a = A[1:- 1]
    b = -A[0:- 1]-A[1:]
    m = x.shape[0]
    D = np.diag(b, 0) + np.diag(a, -1) + np.diag(a, 1)
    f = -h**2*np.ones_like(x).reshape(-1, 1)
    u1 = np.matmul(np.linalg.inv(D), f)
    u = np.ones(m + 2)
    u[1:-1] = u1.flatten() + 1
    # np.save('1d_FDM/1d_q4_%.3f.npy' % eps, u)


if __name__ == '__main__':
    h = 0.0001
    eps_all = [0.5,0.1,0.05,0.01,0.005,0.002]
    for eps in eps_all:
        exam1_fdm(h, eps)
        exam2_fdm(h, eps)
        exam3_fdm(h, eps)
        exam4_fdm(h, eps)
