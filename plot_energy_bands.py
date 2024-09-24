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

k_x_values = np.linspace(-np.pi, np.pi, 1000)
k_y_values = [0]
phi_x = [0]
phi_y = [0]
w_s = 10
w_S = 20
Delta = 0.2
mu = -40
B_x = 0.02
B_y = 0
Lambda = 0.56
w_1 = 0.8

#%% Plot energy bands

E = get_energy(k_x_values, k_y_values, phi_x, phi_y, w_s, w_S, mu, Delta, B_x,
               B_y, Lambda, w_1)

fig, ax = plt.subplots()
ax.plot(k_x_values, E[:, 0, 0, 0, :])
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$E$")
ax.set_title(f"mu={mu}; k_y=0; B_x={B_x}; B_y={B_y}; w_1={w_1}")