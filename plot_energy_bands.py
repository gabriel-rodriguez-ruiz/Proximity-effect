# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:44:11 2024

@author: Gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_y, tau_x
import scipy
from functions import get_energy
    
#%% Parameters
L_x = 10
L_y = 10
k_x_values = np.pi*np.arange(-L_x, L_x)/L_x
k_y_values = np.pi*np.arange(-L_y, L_y)/L_y
k_y = [0]
phi_x = [0]
phi_y = [0]
w_s = 10#10
w_S = 20#20
Delta_s = 0
Delta_S = 0.2
mu = -39
B_x = 0
B_y = 0
Lambda = 0.56
w_1 = [3.4]

#%% Plot energy bands

fig, ax = plt.subplots()
for w in w_1:
    for i in range(8):
        L_x = 1000
        k_x = np.pi*np.arange(-L_x, L_x)/L_x
        Energy = get_energy(k_x, k_y, phi_x, phi_y, w_s, w_S,
                            mu, Delta_s, Delta_S, B_x,
                            B_y, Lambda, w)
        ax.plot(k_x, Energy[:, 0, 0, 0, i] , label=f"{w}")

fig.suptitle(r"$\lambda=$" + f"{np.round(Lambda,2)}"
             +r"; $\Delta=$" + f"{Delta_S}"
             + r"; $\mu=$"+f"{mu}"
             +r"; $w_s=$"+f"{w_s}" + r"; $w_S=$"+ f"{w_S}"
             +r"; $B_y=$"+f"{np.round(B_y, 2)}"
             +r"; $w_1=$" + f"{w_1}")
plt.tight_layout()
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$E(k_x=0, k_y)$")
# plt.legend()