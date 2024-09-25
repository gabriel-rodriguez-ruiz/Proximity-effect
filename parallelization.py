# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:37:26 2024

@author: Gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from pathlib import Path
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_y, tau_x
import scipy
import matplotlib.pyplot as plt
from functions import get_superconducting_density

    
if __name__ == "__main__":
    L_x = 100
    L_y = 100
    w_s = 0#10
    w_S = 10#20
    w_1 = 0
    Delta = 0.2 # 0.2 ###############Normal state
    mu = -40#2*(20*Delta-2*w_0)
    theta = np.pi/2
    Lambda = 0.56#5*Delta/np.sqrt((4*w_0 + mu)/w_0)/2
    h = 1e-2
    k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
    k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
    n_cores = 10
    params = {"L_x": L_x, "L_y": L_y, "w_s": w_s,
              "mu": mu, "Delta": Delta, "theta": theta,
               "Lambda": Lambda,
              "h": h , "k_x_values": k_x_values,
              "k_y_values": k_y_values, "h": h,
              "w_S": w_S, "w_1":w_1}
    
    def integrate(B):
        n = np.zeros(3)
        B_x = B * np.cos(theta)
        B_y = B * np.sin(theta)
        n[0], n[1], n[2] = get_superconducting_density(L_x, L_y, w_s, w_S, mu, Delta, B_x,
                       B_y, Lambda, w_1, h)
        return n
    
    B_values = np.linspace(0, 3*Delta, 10)
    with multiprocessing.Pool(n_cores) as pool:
        results_pooled = pool.map(integrate, B_values)
    n_B_y = np.array(results_pooled)
    
    data_folder = Path("Data/")
    name = f"n_By_mu_{mu}_L={L_x}_h={np.round(h,4)}_B_y_in_({np.min(B_values)}-{np.round(np.max(B_values),3)})_Delta={Delta}_lambda={Lambda}_w_s={w_s}_w_S={w_S}_w_1={w_1}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open , n_B_y=n_B_y, B_values=B_values,
             **params)