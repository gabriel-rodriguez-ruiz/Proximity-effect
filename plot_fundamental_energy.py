#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 08:56:10 2025

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_folder = Path("Data/")
name = "Fundamental_energy_q_x_in_(-0.0126-0)_mu_S_-13.333333333333336_L=500_h=0.0001_B=-0.19_Delta=0.2_lambda=0.5_w_s=10_w_S=3.333_w_1=0.25_points=19.npz"
file_to_open = data_folder / name

data = np.load(file_to_open)
E = data["E"]
# phi_x_values = data["phi_x_values"]
q_x_values = data["q_x_values"]


fig, ax = plt.subplots()
# ax.plot(phi_x_values, E[:, 0, 0])
ax.plot(q_x_values/np.pi, E[:, 0, 0])


# ax.set_xlabel(r"$\phi$")
ax.set_xlabel(r"$q_x/\pi$")

ax.set_ylabel(r"$E$")
ax.legend()
ax.set_title(r"")

fig.tight_layout()
