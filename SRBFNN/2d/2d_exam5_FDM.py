import  numpy as np
import math
import matplotlib.pyplot as plt
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import spsolve


def diff2d(dx,eps):
    """

    """
    x = np.linspace(0,1,int(1/dx)+1)
    n = x.shape[0]
    U = np.zeros((n,n))
    c = 2*math.pi/eps

    #边界条件
    # U[0,:] = 1  #U_0y
    # U[-1,:] = 1#np.cos(x*c)  #U_1y
    # U[:,0] = 1   #U_x0
    # U[:,-1] = 1#np.cos(x*c)  #U_x1

    U[0, :] = 1  # U_0y
    U[-1, :] = 1  # np.cos(x*c)  #U_1y
    U[:, 0] = 1  # U_x0
    U[:, -1] = 1  # np.cos(x*c)  #U_x1

    #右端函数F(x,y)
    # F = np.zeros((n - 2, n - 2))
    # for i in range(n - 2):
    #     for j in range(n - 2):
    #         xi, yj = i * dx, j * dx
    #         F[i, j] = -(xi**2+2*yj**2) * np.cos(c * xi * yj) * (c ** 2) * (xi ** 2 + yj ** 2) - \
    #                   c*np.sin(c * xi * yj) *  (6*xi * yj)
    # F = np.ones((n - 2, n - 2))

    F = 1*np.ones((n - 2, n - 2))
    #a(x,y)
    a1 = np.zeros((n-1,n-2),dtype=np.float32)   #((i+1)*dx,j*dx+0.5*dx),...i=0,1...n-2；j=0,1,2...n-1
    a2 = np.zeros((n-2,n-1),dtype=np.float32)   #(i*dx+0.5*dx,(j+1)*dx),...i=0,1,2...n-1；j=0,1...n-2
    for i in range(n-1):
        for j in range(n-2):
            xi = i*dx+dx
            yj = j*dx+0.5*dx
            a1[i, j] = 2+np.sin(c*(xi+yj))
    for i in range(n-2):
        for j in range(n-1):
            xi = i * dx + 0.5 * dx
            yj = j * dx + dx
            a2[i, j] = 2+np.sin(c*(xi+yj))
    # x1 = np.arange(dx,(n-1)*dx,dx)  #n-2
    # y1 = np.arange(0.5*dx,(n-1)*dx,dx)  #n-1
    # a1 = np.sin(c*np.matmul(y1.reshape(-1, 1), x1.reshape(1, -1)))  #(n-1)*(n-2),,,sin(pi*x*y/eps)
    # a2 = np.sin(c*np.matmul(x1.reshape(-1, 1), y1.reshape(1, -1)))  #(n-2)*(n-1)
    ux0 = U[1:-1, 0]
    ux1 = U[1:-1, -1]
    u0y = U[0, 1:-1]
    u1y = U[-1, 1:-1]

    # print(a1.shape,a2.shape)
    g = a1[0:-1,:] / (dx ** 2)
    h = a1[1:,:] / (dx ** 2)
    d = a2[:, 0: -1] / (dx ** 2)
    f = a2[:, 1: ] / (dx ** 2)
    e = -(g+h+d+f)
    e = e.flatten()

    F = F.flatten()
    F[0: n - 2] = F[0:n -2]-g[0,:]*ux0
    F[- n + 2: ] =F[- n + 2: ] -h[-1,:]*ux1
    F[:: (n - 2)] =F[:: (n - 2)]-d[:, 0]*u0y
    F[n - 3 :: (n - 2)] =F[n - 3:: (n - 2)]-f[:, -1]*u1y

    d = d[:, 1:]
    d = np.concatenate((d,np.zeros((n-2,1))),axis=1)
    d = d.flatten()
    d = d[0:- 1]

    f[:, -1]=0
    f = f.flatten()
    f = f[0:-1]

    h = h[0:- 1,:]
    h = h.flatten()

    g = g[1:,:]
    g = g.flatten()

    data = np.repeat(e.reshape(1,-1),5,axis=0)
    data[1,1:] = f
    data[2,:-1] = d
    data[3,(n-2):] = h
    data[4,:(2-n)] = g
    offsets = np.array([0,1,-1,n-2,2-n])

    A = dia_matrix((data,offsets),shape=(len(e),len(e)))
    # A = np.diag(e, 0) + np.diag(f, 1) + np.diag(d, -1) + np.diag(h, n - 2) + np.diag(g, -n + 2)
    u = spsolve(A,F)
    u = u.reshape(-1,n-2)

    U[1:-1,1:-1] = u
    ux,uy = np.zeros_like(U),np.zeros_like(U)
    ux[1:-1,1:-1] = ((U[2:, 1:-1] - U[:-2, 1:-1]) / (2 * dx))
    uy[1:-1,1:-1] = ((U[1:-1, 2:] - U[1:-1, :-2]) / (2 * dx))
    print(U)
    # plt.figure()
    # ax3 = plt.axes(projection='3d')
    # mesh_x,mesh_y = np.meshgrid(x,x)
    # # U_exact = np.cos(math.pi*mesh_x*mesh_y/eps)
    # ax3.plot_surface(mesh_x, mesh_y, U, rstride=1, cstride=1, cmap='rainbow')
    # ax3.set_xlabel('x')
    # plt.show()

    return U,ux,uy

def show(u):
    print(u.shape)
    x = np.linspace(0, 1,  int(1/dx+1))
    plt.figure()
    plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    ax3 = plt.axes(projection='3d')
    mesh_x, mesh_y = np.meshgrid(x, x)
    ax3.plot_surface(mesh_x, mesh_y, u, rstride=1, cstride=1, cmap='rainbow')
    ax3.set_xlabel('x',fontsize=22)
    ax3.set_title(r'$\varepsilon$=%.3f,FDM解' % e,fontsize=22)
    # plt.savefig('D:/cc/Desktop/tex/SRBF_2d/exam2_FDM2d_%.3f.png' % e)
    plt.show()


if __name__ == '__main__':
    eps = [0.5,0.2,0.1,0.05,0.02,0.01]#0.2,
    dx = 0.0005
    for e in eps:
        U,ux,uy = diff2d(dx,e)
        #plt.plot(U[500,:])
        #plt.show()
        print(ux.shape,uy.shape,U.shape)
        np.save("2d_FDM/2d_exam5_FDM_%.3f.npy" %e,U)
        np.save("2d_FDM/2d_exam5_FDM_x_%.3f.npy" %e,ux)
        np.save("2d_FDM/2d_exam5_FDM_y_%.3f.npy" %e,uy)
        # show(U)
        # print(U1)
        # break
