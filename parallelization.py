# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:37:26 2024

@author: Gabriel
"""

import numpy as np
import multiprocessing
from pathlib import Path
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_y, tau_x
import scipy
from functions import get_superconducting_density

L_x = 1000
L_y = L_x
w_s = 10
w_S = 10/3
w_1 = 0.5   #0.5
Delta_s = 0 # 0.2 ###############Normal state
Delta_S = 0.2
# E_0 = 25.3846  #25.3846
mu_s = -38    #-38
mu_S = 1/3 * mu_s   #-38 + E_0
theta = 0
Lambda = 0.5
h = 1e-4
k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
n_cores = 19
points = 1*n_cores
g = 1/5

params = {"L_x": L_x, "L_y": L_y, "w_s": w_s, "w_S": w_S,
          "mu_s": mu_s, "mu_S": mu_S, "Delta_s": Delta_s, "Delta_S": Delta_S, "theta": theta,
           "Lambda": Lambda,
          "h": h , "k_x_values": k_x_values,
          "k_y_values": k_y_values, "h": h,
          "w_s": w_s, "w_S": w_S, "w_1":w_1,
          "g": g}

def integrate(B):
    n = np.zeros(3)
    B_x = B * np.cos(theta)
    B_y = B * np.sin(theta)
    B_x_S = g * B * np.cos(theta)
    B_y_S = g * B * np.sin(theta)
    n[0], n[1], n[2] = get_superconducting_density(L_x, L_y, w_s, w_S, mu_s, mu_S, Delta_s, Delta_S, B_x,
                                                   B_y, B_x_S, B_y_S, Lambda, w_1, h, q_x_values=[0],
                                                   q_y_values=[0])
    return n

if __name__ == "__main__":
    B_values = np.linspace(0*Delta_S, 2*Delta_S, points)
    with multiprocessing.Pool(n_cores) as pool:
        results_pooled = pool.map(integrate, B_values)
    n_B_y = np.array(results_pooled)
    
    data_folder = Path("Data/")
    name = f"n_By_mu_S_{mu_S}_L={L_x}_h={np.round(h,4)}_B_y_in_({np.min(B_values)}-{np.round(np.max(B_values),3)})_Delta={Delta_S}_lambda={Lambda}_w_s={np.round(w_s, 3)}_w_S={np.round(w_S, 3)}_w_1={np.round(w_1, 3)}_points={points}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open , n_B_y=n_B_y, B_values=B_values,
             **params)
    
    print("\007")