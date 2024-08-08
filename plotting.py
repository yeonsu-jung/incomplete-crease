import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def readmatrix(filepath):
    return np.loadtxt(filepath,delimiter=',')

def plot_half_datasets():
    dir_return = [f for f in os.listdir('results') if f.startswith('data_half_') and f.endswith('.txt')]
    num_datasets = len(dir_return)

    plt.close('all')
    lgd_text = []
    cmap = cm.viridis(np.linspace(0, 1, num_datasets))
    k = 0

    plt.figure(figsize=(9/2.54, 5/2.54))
    for i in [1, 0, 2, 3]:  # MATLAB indices are 1-based, Python is 0-based
        theta_psi_ = readmatrix(os.path.join('results', dir_return[i]))
        az = theta_psi_[:, 0]
        el = theta_psi_[:, 1]

        filename = os.path.splitext(dir_return[i])[0]
        tmp = filename.split('_')[-1]
        lgd_text.append(tmp)

        plt.plot(az, el, 'o', color=cmap[k])
        k += 1

    lgd = plt.legend(lgd_text, loc='best', bbox_to_anchor=(1, 1))
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\psi$')
    # plt.savefig('results/half_crease.png', dpi=600, bbox_inches='tight')
    plt.show()

def plot_inverse_datasets():
    dir_return = [f for f in os.listdir('results') if f.startswith('data_inverse_') and f.endswith('.txt')]
    num_datasets = len(dir_return)

    plt.close('all')
    lgd_text = []
    cmap = cm.viridis(np.linspace(0, 1, 5))
    k = 0

    plt.figure(figsize=(9/2.54, 5/2.54))
    for i in range(num_datasets):
        theta_psi_ = readmatrix(os.path.join('results', dir_return[i]))
        az = theta_psi_[:, 0] - np.min(theta_psi_[:, 0])
        el = theta_psi_[:, 1]

        filename = os.path.splitext(dir_return[i])[0]
        tmp = filename.split('_')[-1]
        lgd_text.append(tmp)

        plt.plot(az, el, 'o', color=cmap[k])
        k += 1

    lgd = plt.legend(lgd_text, loc='best', bbox_to_anchor=(1, 1))
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\psi$')
    plt.savefig('results/inverse_half_crease.png', dpi=600, bbox_inches='tight')
    plt.show()

plot_half_datasets()
plot_inverse_datasets()
