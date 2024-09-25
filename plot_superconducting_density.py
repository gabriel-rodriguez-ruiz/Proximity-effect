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
name = "n_By_mu_-40_L=30_h=0.001_B_y_in_(0.0-0.2)_Delta=0.2_lambda=0.56_w_s=10_w_S=20_w_1=0.8.npz"
file_to_open = data_folder / name

data = np.load(file_to_open)
n_B_y = data["n_B_y"]
B_values = data["B_values"]

fig, ax = plt.subplots()
ax.plot(B_values, n_B_y[:, 1], "o-", label=r"$n_{s,\parallel}$")
ax.plot(B_values, n_B_y[:, 0], "o-", label=r"$n_{s,\perp}$")

ax.set_xlabel(r"$B_y$")
ax.set_ylabel(r"$n_s$")
ax.legend()