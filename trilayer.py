#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 10:17:06 2025

@author: gabriel
"""

import numpy as np
from ZKMBsuperconductor import ZKMBSuperconductor, ZKMBSparseSuperconductor, ZKMBSuperconductorKY
from TrilayerSuperconductor import TrilayerSuperconductor
import scipy
import matplotlib.pyplot as plt
from functions import get_components

# %% Parameters
L_x = 6  # 300
L_y = 1
w_s = 10  # 10
w_S = 10/3  # 20
Delta_s = 0
Delta_S = 0.2  # 0.2  #0.2
E_0 = 25.3846
mu_s = -38  # -38
mu_S = -38 + E_0  # -38 + E_0
theta = np.pi/2
B = 0  # 1.5 * Delta_S
B_x_s = B * np.cos(theta)
B_y_s = B * np.sin(theta)
g = 1/5
B_x_S = g * B * np.cos(theta)
B_y_S = g * B * np.sin(theta)
Lambda = 0.5  # 0.5
w_0_1 = 0.5  # 0.5
w_1_2 = 0.5  # 0.5

k = 10

k_y = 0

H_0 = ZKMBSuperconductor(L_x, L_y, t=w_S, mu=mu_S, Delta_0=Delta_S, Delta_1=0,
                         Lambda=0, B_x=B_x_S, B_y=B_y_S, B_z=0)

H_1 = ZKMBSuperconductor(L_x, L_y, t=w_S, mu=mu_S, Delta_0=Delta_S, Delta_1=0,
                         Lambda=0, B_x=B_x_S, B_y=B_y_S, B_z=0)

H_2 = ZKMBSuperconductor(L_x, L_y, t=w_s, mu=mu_s, Delta_0=0, Delta_1=0,
                         Lambda=Lambda, B_x=B_x_s, B_y=B_y_s, B_z=0)

H = TrilayerSuperconductor(H_0, H_1, H_2, w_0_1, w_1_2)

#%% Real space fundamental energy vs phi

phi_values = np.linspace(0, (L_x-1)*2*np.pi, 50)
Phi_J = np.pi/2
E_phi = np.zeros((len(phi_values), 12*L_x*L_y))

for i, phi in enumerate(phi_values):
    H_0 = ZKMBSuperconductor(L_x, L_y, t=w_S, mu=mu_S, Delta_0=Delta_S*np.exp(1j*Phi_J), Delta_1=0,
                             Lambda=0, B_x=B_x_S, B_y=B_y_S, B_z=0)
    H_1 = ZKMBSuperconductor(L_x, L_y, t=w_S*np.exp(1j*phi/(2*(L_x-1))), mu=mu_S, Delta_0=Delta_S, Delta_1=0,
                             Lambda=0, B_x=B_x_S, B_y=B_y_S, B_z=0)
    H_2 = ZKMBSuperconductor(L_x, L_y, t=w_s*np.exp(-1j*phi/(2*(L_x-1))), mu=mu_s, Delta_0=0, Delta_1=0,
                             Lambda=Lambda, B_x=B_x_s, B_y=B_y_s, B_z=0)
    H = TrilayerSuperconductor(H_0, H_1, H_2, w_0_1, w_1_2)
    E_phi[i, :] = np.linalg.eigvalsh(H.matrix)
negative_energy = np.where(E_phi < 0, E_phi, 0)
fundamental_energy = 1/2*np.sum(negative_energy, axis=(1))

fig, ax = plt.subplots()

ax.plot(phi_values/(2*np.pi), fundamental_energy)
ax.set_xlabel(r"$\varphi/2\pi$")
ax.set_ylabel(r"$E_0(\varphi)$")
ax.set_title(r"$N=$" + f"{L_x}"
             + r"; $\phi_J=" + f"{np.round(Phi_J,3)}$")

#%% Real space fundamental energy vs Phi_J

phi = 2/(2*np.pi)
Phi_J_values = np.linspace(0, 2*np.pi, 50)
# Phi_J_values = np.linspace(-np.pi/100, np.pi/100, 50)

E_Phi_J = np.zeros((len(Phi_J_values), 12*L_x*L_y))

for i, Phi_J in enumerate(Phi_J_values):
    H_0 = ZKMBSuperconductor(L_x, L_y, t=w_S, mu=mu_S, Delta_0=Delta_S*np.exp(1j*Phi_J), Delta_1=0,
                             Lambda=0, B_x=B_x_S, B_y=B_y_S, B_z=0)
    H_1 = ZKMBSuperconductor(L_x, L_y, t=w_S*np.exp(1j*phi/(2*(L_x-1))), mu=mu_S, Delta_0=Delta_S, Delta_1=0,
                             Lambda=0, B_x=B_x_S, B_y=B_y_S, B_z=0)
    H_2 = ZKMBSuperconductor(L_x, L_y, t=w_s*np.exp(-1j*phi/(2*(L_x-1))), mu=mu_s, Delta_0=0, Delta_1=0,
                             Lambda=Lambda, B_x=B_x_s, B_y=B_y_s, B_z=0)
    H = TrilayerSuperconductor(H_0, H_1, H_2, w_0_1, w_1_2)
    E_Phi_J[i, :] = np.linalg.eigvalsh(H.matrix)
negative_energy = np.where(E_Phi_J < 0, E_Phi_J, 0)
fundamental_energy = 1/2*np.sum(negative_energy, axis=(1))

fig, ax = plt.subplots()

ax.plot(Phi_J_values/(2*np.pi), fundamental_energy)
ax.set_xlabel(r"$\phi_J/2\pi$")
ax.set_ylabel(r"$E_0(\phi_J)$")
ax.set_title(r"$N=$" + f"{L_x}"
             + r"; $\varphi=$" + f"{np.round(phi, 3)}")