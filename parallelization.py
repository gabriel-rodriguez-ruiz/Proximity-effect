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

def get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_s, w_S, mu, Delta, B_x, B_y,
                    Lambda, w_1):
    r""" A semiconductor plane over a superconductor plane. The semiconductor
    has spin-orbit coupling and magnetic field.
    
    .. math::
        H_\mathbf{k} = \frac{1}{2} (H_s + H_S + H_{w_1})
        
        H_s = -2w_s\left(\cos(k_x)\cos(\Phi_x) + \cos(k_y)\cos(\Phi_y)\right)
        \tau_z\sigma_0
        - \left(\sin(k_x)\sin(\Phi_x) + \sin(k_y)\sin(\Phi_y)\right)
        \tau_0\sigma_0
        -\mu\tau_z\sigma_0
        + 2\lambda\left(\sin(k_x)\cos(\Phi_x)\tau_z\sigma_y
        + \cos(k_x)\sin(\Phi_x)\tau_0\sigma_y
        - \sin(k_y)\cos(\Phi_y)\tau_z\sigma_x
        - \sin(k_y)\sin(\Phi_y)\tau_0\sigma_x
        - B_x\tau_0\sigma_x - B_y\tau_0\sigma_y \right)
        
        H_S = -2w_S\left(\cos(k_x)\cos(\Phi_x) + \cos(k_y)\cos(\Phi_y)\right)
        \tau_z\sigma_0
        - \left(\sin(k_x)\sin(\Phi_x) + \sin(k_y)\sin(\Phi_y)\right)
        \tau_0\sigma_0
        -\mu\tau_z\sigma_0
        + \Delta \tau_x\sigma_0
        
    """
    H_s = (
        -2*w_s*((np.cos(k_x)*np.cos(phi_x) + np.cos(k_y)*np.cos(phi_y))
               * np.kron(tau_z, sigma_0)
               - (np.sin(k_x)*np.sin(phi_x) + np.sin(k_y)*np.sin(phi_y))
               * np.kron(tau_0, sigma_0)) - mu * np.kron(tau_z, sigma_0)
        + 2*Lambda*(np.sin(k_x)*np.cos(phi_x) * np.kron(tau_z, sigma_y)
                    + np.cos(k_x)*np.sin(phi_x) * np.kron(tau_0, sigma_y)
                    - np.sin(k_y)*np.cos(phi_y) * np.kron(tau_z, sigma_x)
                    - np.cos(k_y)*np.sin(phi_y) * np.kron(tau_0, sigma_x))
        - B_x*np.kron(tau_0, sigma_x) - B_y*np.kron(tau_0, sigma_y)
            ) * 1/2
    H_S = (
        -2*w_S*((np.cos(k_x)*np.cos(phi_x) + np.cos(k_y)*np.cos(phi_y))
               * np.kron(tau_z, sigma_0)
               - (np.sin(k_x)*np.sin(phi_x) + np.sin(k_y)*np.sin(phi_y))
               * np.kron(tau_0, sigma_0)) - mu * np.kron(tau_z, sigma_0)
        + Delta*np.kron(tau_x, sigma_0)
            ) * 1/2
    H_w_1 = -w_1 * np.kron(tau_z, sigma_0)
    H = np.block([
            [H_s, H_w_1],
            [H_w_1, H_S]
        ])
    return H
def get_energy(k_x_values, k_y_values, phi_x_values, phi_y_values, w_s, w_S, mu, Delta, B_x,
               B_y, Lambda, w_1):
    energies = np.zeros((len(k_x_values), len(k_y_values),
                        len(phi_x_values), len(phi_y_values), 8))
    for i, k_x in enumerate(k_x_values):
        for j, k_y in enumerate(k_y_values):
            for k, phi_x in enumerate(phi_x_values):
                for l, phi_y in enumerate(phi_y_values):
                    for m in range(8):
                        H = get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_s, w_S, mu,
                                            Delta, B_x, B_y, Lambda, w_1)
                        energies[i, j, k, l, m] = np.linalg.eigvalsh(H)[m]
    return energies

def get_superconducting_density(L_x, L_y, w_s, w_S, mu, Delta, B_x,
               B_y, Lambda, w_1, h):
    k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
    k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
    phi_x_values = [-h, 0, h]
    phi_y_values = [-h, 0, h]
    E = get_energy(k_x_values, k_y_values, phi_x_values, phi_y_values, w_s,
                   w_S, mu, Delta, B_x, B_y, Lambda, w_1)
    negative_energy = np.where(E<0, E, 0)
    fundamental_energy = 1/2*np.sum(negative_energy, axis=(0, 1, 4))
    #Chequear el w_S abajo
    n_s_xx = 1/w_S * 1/(L_x*L_y) * ( fundamental_energy[2,1] - 2*fundamental_energy[1,1] + fundamental_energy[0,1]) / h**2
    n_s_yy = 1/w_S * 1/(L_x*L_y) * ( fundamental_energy[1,2] - 2*fundamental_energy[1,1] + fundamental_energy[1,0]) / h**2
    n_s_xy = 1/w_S * 1/(L_x*L_y) * ( fundamental_energy[2,2] - fundamental_energy[2,0] - fundamental_energy[0,2] + fundamental_energy[0,0]) / h**2
    return n_s_xx, n_s_yy, n_s_xy

    
if __name__ == "__main__":
    L_x = 4
    L_y = 4
    w_s = 10
    w_S = 20
    w_1 = 0.8
    Delta = 0.2 # 0.2 ###############Normal state
    mu = -39#2*(20*Delta-2*w_0)
    theta = np.pi/2
    Lambda = 0.56#5*Delta/np.sqrt((4*w_0 + mu)/w_0)/2
    h = 1e-2
    k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
    k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
    n_cores = 4
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
    
    B_values = np.linspace(0, 3*Delta, 4)
    with multiprocessing.Pool(n_cores) as pool:
        results_pooled = pool.map(integrate, B_values)
    n_B_y = np.array(results_pooled)
    
    data_folder = Path("Data/")
    name = f"n_By_mu_{mu}_L={L_x}_h={np.round(h,2)}_B_y_in_({np.min(B_values)}-{np.round(np.max(B_values),3)})_Delta={Delta}_lambda={Lambda}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open , n_B_y=n_B_y, B_values=B_values,
             **params)