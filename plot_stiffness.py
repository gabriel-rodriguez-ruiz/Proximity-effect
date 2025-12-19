#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 14:06:45 2025

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_folder = Path("Data/")
name = "Josephson_gauge_L_x=10.npz"
file_to_open = data_folder / name

data = np.load(file_to_open)
fundamental_energy = data["fundamental_energy"]
stiffness = data["stiffness"]
phi_B_values = data["phi_B_values"]
L_x = data["L_x"]
# fig, ax = plt.subplots()
# ax.plot(phi_B_values/(2*np.pi), fundamental_energy[1, :])
# ax.set_xlabel(r"$\phi_B/(2\pi)$")
# ax.set_ylabel(r"$E_0$")

fig, ax = plt.subplots()
ax.plot(phi_B_values/(2*np.pi), stiffness/stiffness[0], label=f"N={L_x}")

ax.set_xlabel(r"$\phi_B/(2\pi)$")
ax.set_ylabel(r"$D_s/D_s(\phi_B=0)$")


#%% stiffness

data_folder = Path("Data/")
name = "Josephson_gauge_L_x=15.npz"
file_to_open = data_folder / name

data = np.load(file_to_open)
fundamental_energy = data["fundamental_energy"]
stiffness = data["stiffness"]
phi_B_values = data["phi_B_values"]
L_x = data["L_x"]

ax.plot(phi_B_values/(2*np.pi), stiffness/stiffness[0], label=f"N={L_x}")

data_folder = Path("Data/")
name = "Josephson_gauge_L_x=20.npz"
file_to_open = data_folder / name

data = np.load(file_to_open)
fundamental_energy = data["fundamental_energy"]
stiffness = data["stiffness"]
phi_B_values = data["phi_B_values"]
L_x = data["L_x"]

ax.plot(phi_B_values/(2*np.pi), stiffness/stiffness[0], label=f"N={L_x}")

data_folder = Path("Data/")
name = "Josephson_gauge_L_x=30.npz"
file_to_open = data_folder / name

data = np.load(file_to_open)
fundamental_energy = data["fundamental_energy"]
stiffness = data["stiffness"]
phi_B_values = data["phi_B_values"]
L_x = data["L_x"]

ax.plot(phi_B_values/(2*np.pi), stiffness/stiffness[0], label=f"N={L_x}")


ax.legend()
plt.grid()

#%% Fundamental energy
fig, ax = plt.subplots()

data_folder = Path("Data/")
name = "Josephson_gauge_L_x=15.npz"
file_to_open = data_folder / name

data = np.load(file_to_open)
fundamental_energy = data["fundamental_energy"]
stiffness = data["stiffness"]
phi_B_values = data["phi_B_values"]
L_x = data["L_x"]

ax.plot(phi_B_values/(2*np.pi), fundamental_energy[1, :]/fundamental_energy[1, 0], label=f"N={L_x}")

data_folder = Path("Data/")
name = "Josephson_gauge_L_x=20.npz"
file_to_open = data_folder / name

data = np.load(file_to_open)
fundamental_energy = data["fundamental_energy"]
stiffness = data["stiffness"]
phi_B_values = data["phi_B_values"]
L_x = data["L_x"]

ax.plot(phi_B_values/(2*np.pi), fundamental_energy[1, :]/fundamental_energy[1, 0], label=f"N={L_x}")

data_folder = Path("Data/")
name = "Josephson_gauge_L_x=30.npz"
file_to_open = data_folder / name

data = np.load(file_to_open)
fundamental_energy = data["fundamental_energy"]
stiffness = data["stiffness"]
phi_B_values = data["phi_B_values"]
L_x = data["L_x"]

ax.plot(phi_B_values/(2*np.pi), fundamental_energy[1, :]/fundamental_energy[1, 0], label=f"N={L_x}")


ax.legend()
plt.grid()
ax.set_xlabel(r"$\phi_B/(2\pi)$")
ax.set_ylabel(r"$E_0/E_0(\phi_B=0)$")

#%% Fundamental energy and stiffness

fig, axs = plt.subplots(2, 1)

data_folder = Path("Data/")
name = "Josephson_gauge_L_x=30.npz"
file_to_open = data_folder / name

data = np.load(file_to_open)
fundamental_energy = data["fundamental_energy"]
stiffness = data["stiffness"]
phi_B_values = data["phi_B_values"]
L_x = data["L_x"]

axs[0].plot(phi_B_values/(2*np.pi), fundamental_energy[1, :], label=f"N={L_x}")
axs[1].plot(phi_B_values/(2*np.pi), stiffness, label=f"N={L_x}")


ax.legend()
plt.grid()
axs[0].set_xlabel(r"$\Phi_B/\Phi_0$")
axs[0].set_ylabel(r"$E_0(\Phi_B, \phi=0)$")
axs[0].grid(True)

axs[1].set_xlabel(r"$\Phi_B/\Phi_0$")
axs[1].set_ylabel(r"$D_s(\Phi_B, \phi=0)$")
axs[1].grid(True)
