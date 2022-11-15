import numpy as np
import matplotlib.pyplot as plt

def show():
    eps = np.array([0.5,0.1,0.05,0.01,0.005,0.002])
    N_exam1 = np.array([17, 38, 66, 186, 367, 750])
    N_exam2 = np.array([17,33,54,159,369,658])
    N_exam3 = np.array([18, 40, 69, 178, 379, 760])
    N_exam4 = np.array([34,73,168,374,605,965])
    plt.rcParams['axes.unicode_minus'] = False
    plt.grid(linestyle="--")
    plt.plot(np.log(eps),np.log(N_exam1),'-o')
    plt.plot(np.log(eps),np.log(N_exam2),'-p')
    plt.plot(np.log(eps),np.log(N_exam3),'-*')
    plt.plot(np.log(eps),np.log(N_exam4),'-h')
    plt.legend(['Example 1','Example 2','Example 3','Example 4'])
    plt.xlabel(r'ln($\varepsilon$)',fontsize=16)
    plt.ylabel('ln($N$)',fontsize=16)
    plt.show()

def cal_order():
    eps_all = np.log(np.array([0.5,0.1,0.05,0.01,0.005,0.002]))
    N_exam1 = np.log(np.array([17, 38, 66, 186, 367, 750])) #-0.693
    N_exam2 = np.log(np.array([17,33,54,159,369,658])) #-0.687
    N_exam3 = np.log(np.array([18, 40, 69, 178, 379, 760]))#-0.683
    N_exam4 = np.log(np.array([34,73,168,374,605,965]))#-0.621

    N = N_exam1
    para = np.polyfit(eps_all,N,deg=1)
    print(para)
    plt.plot(eps_all,N,'-o')
    plt.plot(eps_all,para[0]*eps_all+para[1],'-*')
    plt.show()


if __name__ == "__main__":
    show()
    cal_order()