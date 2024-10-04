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
name = "n_By_mu_-40_L=400_h=0.0001_B_y_in_(0.0-0.1)_Delta=0.2_lambda=0.56_w_s=10_w_S=20_w_1=1.npz"
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
mu = data["mu"]
Lambda = data["Lambda"]
theta = data["theta"]
w_s = data["w_s"]
w_S = data["w_S"]
w_1 = data["w_1"]

ax.set_title(r"$\lambda=$" + f"{Lambda:.2}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{theta:.3}"
             + r"; $\mu$"+f"={mu}"
             +r"; $w_s$"+f"={w_s}"
             +r"; $w_S$"+f"={w_S}"
             +r"; $w_1$"+f"={w_1}")
plt.tight_layout()