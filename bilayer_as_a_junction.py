#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 10:47:57 2025

@author: gabriel
"""

import numpy as np
from junction import Junction, JunctioninZ
from ZKMBsuperconductor import ZKMBSuperconductor, ZKMBSuperconductorFulde, ZKMBSuperconductorKYQYFulde
import matplotlib.pyplot as plt
from BilayerSuperconductor import BilayerSuperconductorKY
from pathlib import Path

L_x = 10  # number of sites, L_x-1 cells
L_y = 1
t_x_s = 1 #  10
t_y_s = 1 #1   #10
t_x_S = 1  #10/3
t_y_S = 1 #1   #10/3
mu_s = -1.6 * t_x_s
mu_S = mu_s  # 1/3 * mu_s
Delta_S = 2   #0.2
Delta_s = 1
Lambda = 0
B_x = 0
B_y = 0
B_z = 0

t_J = 1  #0.5
phi_J = 0 * np.ones(L_y)

params = {"L_x": L_x, "L_y": L_y,
          "t_x_s": t_x_s, "t_y_s": t_y_s,
          "t_x_S": t_x_S, "t_y_S": t_y_S,
          "mu_s": mu_s, "mu_S": mu_S, "Delta_S": Delta_S,
          "Delta_s": Delta_s, "Lambda": Lambda,
          "B_x": B_x, "B_y": B_y, "B_z": B_z}

#%% Josephson gauge with periodic boundary conditions in y

phi_B_values = np.linspace(0, 4*2*np.pi, 50)
k_y_values = np.linspace(0, 2*np.pi, 10)
# k_y_values = [0]
h = 1e-2
q_y_values = [-h, 0, h]
E_q_y_phi_B_k_y = np.zeros((3, len(phi_B_values), len(k_y_values), 8*L_x))

for i, q_y in enumerate(q_y_values):
    for j, phi_B in enumerate(phi_B_values):
        print(phi_B)
        for l, k_y in enumerate(k_y_values):
            # phi_J = phi_B * np.linspace(0, 1, L_x)   # np.linspace(0, 1 , L_x+1)
            phi_J = phi_B * np.linspace(1, L_x, L_x)/L_x
            S_1 = ZKMBSuperconductorKYQYFulde(k=k_y, q=q_y, L_x=L_x, t_x=t_x_S,
                                     t_y=t_y_S, mu=mu_S,
                                     Delta_0=Delta_S*np.ones(L_x+1), Delta_1=0, Lambda=0,
                                     B_x=B_x, B_y=B_y, B_z=B_z)
            S_2 = ZKMBSuperconductorKYQYFulde(k=k_y, q=q_y, L_x=L_x, t_x=t_x_s,
                                     t_y=t_y_s, mu=mu_s,
                                     Delta_0=Delta_s*np.ones(L_x+1), Delta_1=0, Lambda=0,  # Delta_0=np.zeros(L_x+1)
                                     B_x=B_x, B_y=B_y, B_z=B_z)
            J = JunctioninZ(S_1, S_2, t_J, phi_J)         # I put 2*phi_J because because the Josephson effect is 2pi periodic and not 4pi like in TRITOPS
            E_q_y_phi_B_k_y[i, j, l, :] = np.linalg.eigvalsh(J.matrix.toarray())

negative_energies = np.where(E_q_y_phi_B_k_y<0, E_q_y_phi_B_k_y, 0)
fundamental_energy = np.sum(negative_energies, axis=(2, 3))

stiffness = (fundamental_energy[2,:] - 2*fundamental_energy[1, :] + fundamental_energy[0, :])/h**2

#%% Plot energy bands

fig, ax = plt.subplots()

k_y_index = 0
for i in range(8*L_x):
    ax.plot(phi_B_values/(2*np.pi), E_q_y_phi_B_k_y[1, :, k_y_index, i])

ax.set_xlabel(r"$\phi_B/(2\pi)$")
ax.set_ylabel(r"$E$")
ax.set_title(f"k_y={np.round(k_y_values[k_y_index], 3)}")
plt.grid()



#%% Plot stiffness

fig, ax = plt.subplots()
ax.plot(phi_B_values/(2*np.pi), stiffness)

ax.set_xlabel(r"$\phi_B/(2\pi)$")
ax.set_ylabel(r"$D_s$")
ax.set_title(r"N=" + f"{L_x}")
ax.legend()
plt.grid()

fig, ax = plt.subplots()
ax.plot(phi_B_values/(2*np.pi), fundamental_energy[1, :])
ax.set_xlabel(r"$\phi_B/(2\pi)$")
ax.set_ylabel(r"$E_0$")
ax.set_title(r"N=" + f"{L_x}")

plt.grid()

#%% Save stiffness

data_folder = Path("Data/")
name = f"Josephson_gauge_L_x={L_x}.npz"
file_to_open = data_folder / name
np.savez(file_to_open , fundamental_energy=fundamental_energy, phi_B_values=phi_B_values,
         stiffness=stiffness,
         **params)

#%% Fulde gauge with periodic boundary conditions in y

phi_B_values = np.linspace(0, 2*L_x*2*np.pi, 50)
k_y_values = np.linspace(0, 2*np.pi, 10)
# k_y_values = [0]
h = 1e-2
q_y_values = [-h, 0, h]
E_q_y_phi_B_k_y = np.zeros((3, len(phi_B_values), len(k_y_values), 8*L_x))

for i, q_y in enumerate(q_y_values):
    for j, phi_B in enumerate(phi_B_values):
        print(phi_B)
        Delta_x_S = Delta_S * np.exp(-1j * phi_B/2 * np.linspace(0, 1 , L_x))  # Delta_x_S = Delta_S * np.exp(-1j * phi_B/2 * np.linspace(0, 1 , L_x+1))
        Delta_x_s = Delta_s * np.exp(1j * phi_B/2 * np.linspace(0, 1 , L_x))  # Delta_x_s = Delta_s * np.exp(1j * phi_B/2 * np.linspace(0, 1 , L_x+1))
        for l, k_y in enumerate(k_y_values):
            #phi_J = phi_B * np.linspace(0, 1 , L_x+1)
            S_1 = ZKMBSuperconductorKYQYFulde(k=k_y, q=q_y, L_x=L_x, t_x=t_x_S*np.exp(1j*phi_B/(4*(L_x-1))),   # t_x=t_x_S*np.exp(1j*phi_B/(4*L_x))
                                     t_y=t_y_S, mu=mu_S,
                                     Delta_0=Delta_x_S, Delta_1=0, Lambda=0,
                                     B_x=B_x, B_y=B_y, B_z=B_z)
            S_2 = ZKMBSuperconductorKYQYFulde(k=k_y, q=q_y, L_x=L_x, t_x=t_x_s*np.exp(-1j*phi_B/(4*(L_x-1))),   # t_x_s*np.exp(-1j*phi_B/(4*L_x))
                                     t_y=t_y_s, mu=mu_s,
                                     Delta_0=Delta_x_s, Delta_1=0, Lambda=0,  # Delta_0=np.zeros(L_x+1)
                                     B_x=B_x, B_y=B_y, B_z=B_z)
            H = BilayerSuperconductorKY(S_1, S_2, t_J)
            E_q_y_phi_B_k_y[i, j, l, :] = np.linalg.eigvalsh(H.matrix)

negative_energies = np.where(E_q_y_phi_B_k_y<0, E_q_y_phi_B_k_y, 0)
fundamental_energy = np.sum(negative_energies, axis=(2, 3))

stiffness = (fundamental_energy[2,:] - 2*fundamental_energy[1, :] + fundamental_energy[0, :])/h**2

#%% Plot energy bands

fig, ax = plt.subplots()

k_y_index = 0
for i in range(8*L_x):
    ax.plot(phi_B_values/(2*np.pi), E_q_y_phi_B_k_y[1, :, k_y_index, i])

ax.set_xlabel(r"$\phi_B/(2\pi)$")
ax.set_ylabel(r"$E$")
ax.set_title(f"k_y={np.round(k_y_values[k_y_index], 3)}")
plt.grid()


#%% Plot fundamental energy in the Fulde gauge

fig, ax = plt.subplots()
ax.plot(phi_B_values/(2*np.pi), stiffness)

ax.set_xlabel(r"$\phi_B/(2\pi)$")
ax.set_ylabel(r"$D_s$")
ax.set_title(r"N=" + f"{L_x}")
plt.grid()

fig, ax = plt.subplots()
ax.plot(phi_B_values/(2*np.pi), fundamental_energy[1, :])
ax.set_xlabel(r"$\phi_B/(2\pi)$")
ax.set_ylabel(r"$E_0$")
ax.set_title(r"N=" + f"{L_x}")
plt.grid()

