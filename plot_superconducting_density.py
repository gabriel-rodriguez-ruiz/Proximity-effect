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
name = "n_By_mu_-40_L=400_h=0.01_B_y_in_(0.0-0.6)_Delta=0.2_lambda=0.56.npz"
file_to_open = data_folder / name

data = np.load(file_to_open)
n_B_y = data["n_B_y"]
B_values = data["B_values"]

fig, ax = plt.subplots()
ax.plot(B_values, n_B_y[:, 1], "o")
ax.set_xlabel(r"$B_y$")
ax.set_ylabel(r"$n_s^{yy}$")