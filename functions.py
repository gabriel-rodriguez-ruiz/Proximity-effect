# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 09:45:20 2024

@author: Gabriel
"""
import numpy as np
from hamiltonian import get_Hamiltonian
from pauli_matrices import tau_0, sigma_0

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

def get_Green_function(omega, k_x_values, k_y_values, w_s, w_S, mu, Delta, B_x, B_y,
                    Lambda, w_1):
    G = np.zeros((len(k_x_values), len(k_y_values),
                  4, 4), dtype=complex)
    for i, k_x in enumerate(k_x_values):
        for j, k_y in enumerate(k_y_values):
            for l in range(4):
                for m in range(4):
                    G[i, j, l, m] = np.linalg.inv(omega*np.kron(tau_0, sigma_0)
                                                  - get_Hamiltonian(k_x, k_y, 0, 0, w_s, w_S, mu, Delta, B_x, B_y,
                                                                      Lambda, w_1) 
                                                  )[l, m]                
    return G

def get_DOS(omega, eta, L_x, L_y, w_s, w_S, mu, Delta, B_x, B_y,
                    Lambda, w_1):
    k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
    k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
    G = get_Green_function(omega+1j*eta, k_x_values, k_y_values, w_s, w_S, mu, Delta, B_x, B_y,
                        Lambda, w_1)
    return 1/(L_x*L_y) * 1/np.pi*np.sum(-np.imag(G), axis=(0,1))

