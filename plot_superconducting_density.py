#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 14:07:15 2024

@author: gabriel
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_folder = Path("Data/")
name = "n_By_mu_S_-12.615400000000001_L=100_h=0.01_B_y_in_(0.0-0.4)_Delta=0.2_lambda=0.5_w_s=10_w_S=3.333_w_1=0.5_points=19.npz"
file_to_open = data_folder / name

data = np.load(file_to_open)
n_B_y = data["n_B_y"]
B_values = data["B_values"]

fig, ax = plt.subplots()
ax.plot(B_values, n_B_y[:, 0], "o-", label=r"$n_{s,\perp}$")
ax.plot(B_values, n_B_y[:, 1], "o-", label=r"$n_{s,\parallel}$")

ax.set_xlabel(r"$B_y$")
ax.set_ylabel(r"$n_s$")
ax.legend()

B_values = data["B_values"]
Delta = data["Delta_S"]
mu_s = data["mu_s"]
Lambda = data["Lambda"]
theta = data["theta"]
w_s = data["w_s"]
w_S = data["w_S"]
w_1 = data["w_1"]
L = data["L_x"]
# field_in_S = data["field_in_S"]

ax.set_title(r"$\lambda=$" + f"{Lambda:.2}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{theta}"
             + r"; $\mu_s$"+f"={mu_s}"
             +r"; $w_s$"+f"={w_s}"
             +r"; $w_S$"+f"={w_S}"
             +r"; $w_1$"+f"={w_1}"
             +r"; $L$" + f"={L}")

fig.tight_layout()
