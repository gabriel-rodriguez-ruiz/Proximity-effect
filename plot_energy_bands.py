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
L_x = 50
L_y = 50
k_x_values = np.pi*np.arange(-L_x, L_x)/L_x
k_y_values = np.pi*np.arange(-L_y, L_y)/L_y
k_y = [0]
phi_x = [0]
phi_y = [0]
w_s = 10
w_S = 20
Delta = 0.2
mu = -39
B_x = 0
B_y = 0.4
Lambda = 0.56
w_1 = 0.8

#%% Plot energy bands

E = get_energy(k_x_values, k_y_values, phi_x, phi_y, w_s, w_S, mu, Delta, B_x,
               B_y, Lambda, w_1)

fig, ax = plt.subplots(1, 2)
ax1 = ax[0]
ax2 = ax[1]
for i in range(8):
    Energy = get_energy(k_x_values, k_y, phi_x, phi_y, w_s, w_S, mu, Delta, B_x,
                   B_y, Lambda, w_1)
    ax1.plot(k_x_values, Energy[:, 0, 0, 0, i] )

ax1.set_xlabel(r"$k_x$")
ax1.set_ylabel(r"$E(k_x,k_y=$"+f"{np.round(k_y[0],2)})")
X, Y = np.meshgrid(k_x_values, k_y_values)
C1 = ax2.contour(Y, X, E[:,:,0,0,3]>0, 0, colors="C3") #notice the inversion of X and Y
C2 = ax2.contour(Y, X, E[:,:,0,0,4]<0, 0, colors="C4")
C3 = ax2.contour(Y, X, E[:,:,0,0,2], 10, colors="C2")
ax2.clabel(C1, inline=True, fontsize=10)
ax2.clabel(C2, inline=True, fontsize=10)
ax2.clabel(C3, inline=True, fontsize=10)
ax2.set_xlabel(r"$k_x$")
ax2.set_ylabel(r"$k_y$")

fig.suptitle(r"$\lambda=$" + f"{np.round(Lambda,2)}"
             +r"; $\Delta=$" + f"{Delta}"
             + r"; $\mu=$"+f"{mu}"
             +r"; $w_s=$"+f"{w_s}" + r"; $w_S=$"+ f"{w_S}"
             +r"; $B_y=$"+f"{np.round(B_y, 2)}")
plt.tight_layout()
