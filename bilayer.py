#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 14:17:00 2025

@author: gabriel
"""

import numpy as np
from ZKMBsuperconductor import ZKMBSuperconductor, ZKMBSparseSuperconductor, ZKMBSuperconductorKY, ZKMBSuperconductorKYQY, ZKMBSuperconductorFulde
from BilayerSuperconductor import BilayerSuperconductor, SparseBilayerSuperconductor, SparseBilayerSuperconductorPeriodicInY, BilayerSuperconductorKY
import scipy
import matplotlib.pyplot as plt
from functions import get_components

#%% Parameters

L_x = 8  # 300
L_y = 1
w_s = 5#10  # 10
w_S = 5 #10/3  # 20
Delta_s = 0
Delta_S = 2  # 0.2  #0.2
E_0 = 0#25.3846
mu_s = -10  # -38
mu_S = -10 + E_0  # -38 + E_0
theta = 0
B = 0  #2*Delta_S  # 1.5 * Delta_S
B_x_s = B * np.cos(theta)
B_y_s = B * np.sin(theta)
g = 1/5
B_x_S = g * B * np.cos(theta)
B_y_S = g * B * np.sin(theta)
Lambda = 0#0.5  # 0.5
w_1 = 0.5#0.5  # 0.5

k = 10

k_y = 0

H_0 = ZKMBSuperconductor(L_x, L_y, t_x=w_S, t_y=w_S, mu=mu_S, Delta_0=Delta_S, Delta_1=0,
                         Lambda=0, B_x=B_x_S, B_y=B_y_S, B_z=0)

# H_0 = ZKMBSparseSuperconductor(L_x, L_y, t=w_S, mu=mu_S, Delta_0=Delta_S, Delta_1=0,
#                          Lambda=0, B_x=B_x_S, B_y=B_y_S, B_z=0)

# H_0 = ZKMBSuperconductorKY(k=k_y, L_x=L_x, t=w_S, mu=mu_S, Delta_0=Delta_S, Delta_1=0,
#                          Lambda=0, B_x=B_x_S, B_y=B_y_S, B_z=0)

H_1 = ZKMBSuperconductor(L_x, L_y, t_x=w_s, t_y=w_s, mu=mu_s, Delta_0=0, Delta_1=0,
                         Lambda=Lambda, B_x=B_x_s, B_y=B_y_s, B_z=0)

# H_1 = ZKMBSparseSuperconductor(L_x, L_y, t=w_s, mu=mu_s, Delta_0=0, Delta_1=0,
#                                Lambda=Lambda, B_x=B_x_s, B_y=B_y_s, B_z=0)

# H_1 = ZKMBSuperconductorKY(k=k_y, L_x=L_x, t=w_s, mu=mu_s, Delta_0=0, Delta_1=0,
#                                Lambda=Lambda, B_x=B_x_s, B_y=B_y_s, B_z=0)

H = BilayerSuperconductor(H_0, H_1, w_1)

# H = SparseBilayerSuperconductor(H_0, H_1, w_1)

# H = SparseBilayerSuperconductorPeriodicInY(H_0, H_1, w_1)

# H = BilayerSuperconductorKY(H_0, H_1, w_1)

#%% Real space fundamental energy

phi_B_values = np.linspace(0, (L_x-1)*2*np.pi, 50)

E_phi_B = np.zeros((len(phi_B_values), 8*L_x*L_y))

for i, phi_B in enumerate(phi_B_values):
    phi_J = phi_B * np.linspace(0, 1 , L_x)
    H_0 = ZKMBSuperconductor(L_x=L_x, L_y=L_y, t_x=w_S, t_y=w_S, mu=mu_S,
                             Delta_0=Delta_S, Delta_1=0,
                             Lambda=0, B_x=B_x_S, B_y=B_y_S, B_z=0)
    H_1 = ZKMBSuperconductor(L_x=L_x, L_y=L_y, t_x=w_s, t_y=w_s,
                             mu=mu_s, Delta_0=0, Delta_1=0,
                             Lambda=Lambda, B_x=B_x_s, B_y=B_y_s, B_z=0)
    H = BilayerSuperconductor(H_0, H_1, w_1*np.exp(1j*phi_J))
    E_phi_B[i, :] = np.linalg.eigvalsh(H.matrix)

negative_energy = np.where(E_phi_B < 0, E_phi_B, 0)
fundamental_energy = 1/2*np.sum(negative_energy, axis=(1))

fig, ax = plt.subplots()

ax.plot(phi_B_values/(2*np.pi), fundamental_energy)
ax.set_xlabel(r"$\varphi/2\pi$")
ax.set_ylabel(r"$E_0(\varphi)$")
ax.set_title(r"$N=$" + f"{L_x}")

# ax.plot(phi_values/(2*np.pi), E_phi[:, 0])

#%%

fig, ax = plt.subplots()
for i in range(8*L_x*L_y):
    ax.plot(phi_values/(2*np.pi), E_phi[:, i])

ax.set_xlabel(r"$\varphi/2\pi$")
ax.set_ylabel(r"$E_0(\varphi)$")
ax.set_title(r"$N=$" + f"{L_x}")
plt.grid()


#%% Real space stiffness
h = 1e-2   # 1e-1 #*np.pi #* 2*(L_x-1)
q_values = [-h, 0, h]
phi_values = np.linspace(0, (L_y-1)*2*np.pi, 100)

E_phi_q = np.zeros((3, len(phi_values), 8*L_x*L_y))

for j, q in enumerate(q_values):
    for i, phi in enumerate(phi_values):
        H_0 = ZKMBSuperconductorFulde(L_x, L_y, t_x=w_S, t_y=w_S*np.exp(1j*(phi+q)/(2*L_y)), mu=mu_S, Delta_0=Delta_S*np.exp(1j * phi * np.linspace(0, 1 , L_y+1)), Delta_1=0,
                                 Lambda=0, B_x=B_x_S, B_y=B_y_S, B_z=0)
        H_1 = ZKMBSuperconductorFulde(L_x, L_y, t_x=w_s, t_y=w_s*np.exp(-1j*(phi)/(2*L_y)), mu=mu_s, Delta_0=np.zeros(L_y+1), Delta_1=0,
                                 Lambda=Lambda, B_x=B_x_s, B_y=B_y_s, B_z=0)
        H = BilayerSuperconductor(H_0, H_1, w_1)
        E_phi_q[j, i, :] = np.linalg.eigvalsh(H.matrix)


negative_energy = np.where(E_phi_q < 0, E_phi_q, 0)
fundamental_energy = 1/2*np.sum(negative_energy, axis=(2))
stiffness = (fundamental_energy[2,:] - 2*fundamental_energy[1, :] + fundamental_energy[0, :])/h**2

fig, ax = plt.subplots()

# ax.plot(phi_values/(2*np.pi), fundamental_energy[0, :])
# ax.plot(phi_values/(2*np.pi), fundamental_energy[1, :])
# ax.plot(phi_values/(2*np.pi), fundamental_energy[2, :])

ax.plot(phi_values/(2*np.pi), stiffness)
# ax.plot(phi_values/(2*np.pi), np.gradient(np.gradient(fundamental_energy[1, :])))
# ax.plot(phi_values/(2*np.pi), np.gradient(fundamental_energy[1, :]))

ax.set_xlabel(r"$\varphi/2\pi$")
ax.set_ylabel(r"$D_s(\varphi)$")
ax.set_title(r"$N=$" + f"{L_y}")



#%% Levels for different sizes

fig, ax = plt.subplots()
N_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# N_values = [8]    #np.arange(2, 10, 1)
n = 3

for j, N in enumerate(N_values):
    L_x = N
    phi_values = np.linspace(0, (L_x-1)*2*np.pi, 50)

    E_phi = np.zeros((len(phi_values), 8*L_x*L_y))

    for i, phi in enumerate(phi_values):
        H_0 = ZKMBSuperconductor(L_x, L_y, t=w_S*np.exp(1j*phi/(2*(L_x-1))), mu=mu_S, Delta_0=Delta_S, Delta_1=0,
                                 Lambda=0, B_x=B_x_S, B_y=B_y_S, B_z=0)
        H_1 = ZKMBSuperconductor(L_x, L_y, t=w_s*np.exp(-1j*phi/(2*(L_x-1))), mu=mu_s, Delta_0=0, Delta_1=0,
                                 Lambda=Lambda, B_x=B_x_s, B_y=B_y_s, B_z=0)
        H = BilayerSuperconductor(H_0, H_1, w_1)
        E_phi[i, :] = np.linalg.eigvalsh(H.matrix)
    negative_energy = np.where(E_phi < 0, E_phi, 0)
    fundamental_energy = 1/2*np.sum(negative_energy, axis=(1))
    # ax.plot(phi_values/(2*np.pi), E_phi[:, n], label=f"N={N}")
    ax.plot(phi_values/(2*np.pi), E_phi[:, n]/E_phi[0, n], label=f"N={N}")

    # ax.plot(phi_values/(2*np.pi), E_phi[:, i+2]/E_phi[0, i+2], linestyle="dashed")
    ax.set_title(f"n={n}")
    label_index = np.where(np.abs(E_phi[:, n]/E_phi[0, n]-1)==np.max(np.abs(E_phi[:, n]/E_phi[0, n]-1)))[0][0]
    ax.annotate(f"{N}", xy=(phi_values[label_index]/(2*np.pi), (E_phi[label_index, n]/E_phi[0, n])),
                color=ax.lines[j].get_color(),
                size=14, va="center")
plt.legend()
ax.set_ylabel(r"$E_n(\phi)/E_n(\phi=0)$")
ax.set_xlabel(r"$\phi/2\pi$")

#%%

fig, ax = plt.subplots()
n_values = np.arange(0, 8, 1)
N = L_x
for j, n in enumerate(n_values):
    phi_values = np.linspace(0, (L_x-1)*2*np.pi, 50)

    E_phi = np.zeros((len(phi_values), 8*L_x*L_y))

    for i, phi in enumerate(phi_values):
        H_0 = ZKMBSuperconductor(L_x, L_y, t=w_S*np.exp(1j*phi/(2*(L_x-1))), mu=mu_S, Delta_0=Delta_S, Delta_1=0,
                                 Lambda=0, B_x=B_x_S, B_y=B_y_S, B_z=0)
        H_1 = ZKMBSuperconductor(L_x, L_y, t=w_s*np.exp(-1j*phi/(2*(L_x-1))), mu=mu_s, Delta_0=0, Delta_1=0,
                                 Lambda=Lambda, B_x=B_x_s, B_y=B_y_s, B_z=0)
        H = BilayerSuperconductor(H_0, H_1, w_1)
        E_phi[i, :] = np.linalg.eigvalsh(H.matrix)
    negative_energy = np.where(E_phi < 0, E_phi, 0)
    fundamental_energy = 1/2*np.sum(negative_energy, axis=(1))
    # ax.plot(phi_values/(2*np.pi), E_phi[:, n], label=f"N={N}")
    ax.plot(phi_values/(2*np.pi), E_phi[:, n], label=f"N={N}")

    # ax.plot(phi_values/(2*np.pi), E_phi[:, i+2]/E_phi[0, i+2], linestyle="dashed")
    ax.set_title(f"N={N}")
    label_index = np.where(np.abs(E_phi[:, n]/E_phi[0, n]-1)==np.max(np.abs(E_phi[:, n]/E_phi[0, n]-1)))[0][0]
    ax.annotate(f"{N}", xy=(phi_values[label_index]/(2*np.pi), (E_phi[label_index, n]/E_phi[0, n])),
                color=ax.lines[j].get_color(),
                size=14, va="center")
plt.legend()
ax.set_ylabel(r"$E_n(\phi)/E_n(\phi=0)$")
ax.set_xlabel(r"$\phi/2\pi$")