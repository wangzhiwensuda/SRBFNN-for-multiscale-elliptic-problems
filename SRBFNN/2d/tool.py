import numpy as np
import matplotlib.pyplot as plt

def show_2d_N():
    eps = np.array([0.5, 0.2, 0.1, 0.05, 0.02, 0.01])
    Ne_exam1 = np.array([42, 146, 531, 2121, 6272, 16326])
    Ne_exam2 = np.array([48, 152, 538, 2275, 6362, 17285])

    plt.rcParams['axes.unicode_minus'] = False
    plt.grid(linestyle="--")
    plt.plot(np.log(eps), np.log(Ne_exam1), '-o')
    plt.plot(np.log(eps), np.log(Ne_exam2), '-*')

    plt.legend(['Example 5', 'Example 6'])
    plt.xlabel(r'ln($\varepsilon$)', fontsize=16)
    plt.ylabel('ln($N$)', fontsize=16)
    plt.show()


def get_order():
    eps_all = -np.log(np.array([0.5,0.2,0.1,0.05,0.02,0.01]))#
    N_exam5 = np.log(np.array([42, 146, 531, 2121, 6272, 16326])) #1.560
    N_exam6 = np.log(np.array([48, 152, 538, 2275, 6362, 17285])) #1.544
    N_exam8 = np.log(np.array([227,753,1988,4673]))

    para = np.polyfit(eps_all,N_exam5,deg=1)
    print(para)
    plt.plot(eps_all,N_exam5,'-o')
    plt.plot(eps_all,para[0]*eps_all+para[1],'-*')
    plt.show()





if __name__ == '__main__':
    show_2d_N()
    get_order()
