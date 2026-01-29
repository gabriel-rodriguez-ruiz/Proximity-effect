# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 09:45:20 2024

@author: Gabriel
"""
import numpy as np
from hamiltonian import get_Hamiltonian, get_Hamiltonian_in_polars
from pauli_matrices import tau_0, sigma_0

def get_energy(k_x_values, k_y_values, phi_x_values, phi_y_values, w_s, w_S,
               mu_s, mu_S, Delta_s, Delta_S, B_x,
               B_y, B_x_S, B_y_S, Lambda, w_1, q_B_x_values, q_B_y_values,
               q_x_values, q_y_values):
    energies = np.zeros((len(k_x_values), len(k_y_values),
                        len(phi_x_values), len(phi_y_values),
                        len(q_B_x_values), len(q_B_y_values), 
                        len(q_x_values), len(q_y_values),
                        8))
    for i, k_x in enumerate(k_x_values):
        for j, k_y in enumerate(k_y_values):
            for p, phi_x in enumerate(phi_x_values):
                for l, phi_y in enumerate(phi_y_values):
                    for m, q_B_x in enumerate(q_B_x_values):
                        for n, q_B_y in enumerate(q_B_y_values):
                            for r, q_x in enumerate(q_x_values):
                                for s, q_y in enumerate(q_y_values):
                                    H = get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_s, w_S,
                                                        mu_s, mu_S, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                                                        Lambda, w_1, q_B_x, q_B_y, q_x, q_y)
                                    E = np.linalg.eigvalsh(H)
                                    for o in range(8):
                                        energies[i, j, p, l, m, n, r, s, o] = E[o]
    return energies

def get_energy_in_polars(k_values, theta_values, phi_x_values, phi_y_values, w_s, w_S,
                           mu_s, mu_S, Delta_s, Delta_S, B_x,
                           B_y, B_x_S, B_y_S, Lambda, w_1, q_x_values, q_y_values):
    energies = np.zeros((len(k_values), len(theta_values),
                        len(phi_x_values), len(phi_y_values),
                        len(q_x_values), len(q_y_values), 8))
    for i, k in enumerate(k_values):
        for j, theta in enumerate(theta_values):
            for p, phi_x in enumerate(phi_x_values):
                for l, phi_y in enumerate(phi_y_values):
                    for m, q_x in enumerate(q_x_values):
                        for n, q_y in enumerate(q_y_values):
                            H = get_Hamiltonian_in_polars(k, theta, phi_x, phi_y, w_s, w_S,
                                                mu_s, mu_S, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                                                Lambda, w_1, q_x, q_y)
                            E = np.linalg.eigvalsh(H)
                            for o in range(8):
                                energies[i, j, p, l, m, n, o] = E[o]
    return energies

def get_superconducting_density(L_x, L_y, w_s, w_S, mu_s, mu_S, Delta_s, Delta_S, B_x,
               B_y, B_x_S, B_y_S, Lambda, w_1, h, q_x_values, q_y_values):
    k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
    k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
    phi_x_values = [-h, 0, h]
    phi_y_values = [-h, 0, h]
    E = get_energy(k_x_values, k_y_values, phi_x_values, phi_y_values, w_s,
                   w_S, mu_s, mu_S, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S, Lambda, w_1,
                   q_x_values, q_y_values)
    negative_energy = np.where(E<0, E, 0)
    fundamental_energy = 1/2*np.sum(negative_energy, axis=(0, 1, 4, 5, 6))
    #Chequear el w_S abajo
    n_s_xx = 1/w_S * 1/(L_x*L_y) * ( fundamental_energy[2,1] - 2*fundamental_energy[1,1] + fundamental_energy[0,1]) / h**2
    n_s_yy = 1/w_S * 1/(L_x*L_y) * ( fundamental_energy[1,2] - 2*fundamental_energy[1,1] + fundamental_energy[1,0]) / h**2
    n_s_xy = 1/w_S * 1/(L_x*L_y) * ( fundamental_energy[2,2] - fundamental_energy[2,0] - fundamental_energy[0,2] + fundamental_energy[0,0]) / h**2
    return n_s_xx, n_s_yy, n_s_xy

def get_fundamental_energy(L_x, L_y, w_s, w_S, mu_s, mu_S, Delta_s, Delta_S, B_x,
               B_y, B_x_S, B_y_S, Lambda, w_1, phi_x_values, phi_y_values,
               q_B_x_values, q_B_y_values,
               q_x_values, q_y_values):
    k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
    k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
    E = get_energy(k_x_values, k_y_values, phi_x_values, phi_y_values, w_s,
                   w_S, mu_s, mu_S, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S, Lambda, w_1,
                   q_B_x_values, q_B_y_values, q_x_values, q_y_values)
    negative_energy = np.where(E<0, E, 0)
    # fundamental_energy = 1/2*np.sum(negative_energy, axis=(0, 1, 4, 5, 6))
    fundamental_energy = 1/2*np.sum(negative_energy)
    return fundamental_energy

def get_fundamental_energy_in_polars(L_k, L_theta, w_s, w_S, mu_s, mu_S, Delta_s, Delta_S, B_x,
               B_y, B_x_S, B_y_S, Lambda, w_1, phi_x_values, phi_y_values, q_x_values, q_y_values):
    radius_values = np.linspace(0, np.pi, L_k)
    theta_values = np.linspace(-np.pi/2, 3*np.pi/2, L_theta)
    E = get_energy_in_polars(radius_values, theta_values, phi_x_values, phi_y_values, w_s,
                   w_S, mu_s, mu_S, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S, Lambda, w_1,
                   q_x_values, q_y_values)
    negative_energy = np.where(E<0, E, 0)
    fundamental_energy = 1/2*np.sum(negative_energy)
    return fundamental_energy

def get_components(state, L_x, L_y):
    """
    Get the components of the state: creation_up,
    creation_down, destruction_down, destruction_up for a given
    column state. Returns an array of shape (L_y, L_x)
    """
    destruction_up_0 = state[0:int(4*L_x*L_y):4].reshape((L_x, L_y))
    destruction_down_0 = state[1:int(4*L_x*L_y):4].reshape((L_x, L_y))
    creation_down_0 = state[2:int(4*L_x*L_y):4].reshape((L_x, L_y))
    creation_up_0 = state[3:int(4*L_x*L_y):4].reshape((L_x, L_y))
    destruction_up_1 = state[int(4*L_x*L_y)::4].reshape((L_x, L_y))
    destruction_down_1 = state[int(4*L_x*L_y)+1::4].reshape((L_x, L_y))
    creation_down_1 = state[int(4*L_x*L_y)+2::4].reshape((L_x, L_y))
    creation_up_1 = state[int(4*L_x*L_y)+3::4].reshape((L_x, L_y))
    return (np.flip(destruction_up_0.T, axis=0),
            np.flip(destruction_down_0.T, axis=0),
            np.flip(creation_down_0.T, axis=0),
            np.flip(creation_up_0.T, axis=0),
            np.flip(destruction_up_1.T, axis=0),
            np.flip(destruction_down_1.T, axis=0),
            np.flip(creation_down_1.T, axis=0),
            np.flip(creation_up_1.T, axis=0))