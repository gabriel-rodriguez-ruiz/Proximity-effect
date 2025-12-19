#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 09:29:03 2025

@author: gabriel
"""

import numpy as np
from pauli_matrices import tau_x, sigma_x, tau_z, sigma_0, sigma_y,\
                            tau_0, sigma_z
from hamiltonian import BilayerHamiltonian, SparseBilayerHamiltonian, SparsePeriodicBilayerHamiltonianInY
from ZKMBsuperconductor import ZKMBSuperconductor, ZKMBSuperconductorKY

class BilayerSuperconductivity():
    def __init__(self, w_1: complex):
        self.w_1 = w_1
    def _get_hopping_z(self):
        return -1/2*(self.w_1*np.kron((tau_0+tau_z)/2, sigma_0)  #-1/2
                     - np.conj(self.w_1)*np.kron((tau_0-tau_z)/2, sigma_0)
                     )

class BilayerSuperconductor(BilayerSuperconductivity,
                            BilayerHamiltonian):
    def __init__(self, H_0:ZKMBSuperconductor, H_1:ZKMBSuperconductor, w_1:float):
        BilayerSuperconductivity.__init__(self, w_1)
        self.H_0 = H_0
        self.H_1 = H_1
        BilayerHamiltonian.__init__(self, self.H_0, self.H_1, self._get_hopping_z())
        
class SparseBilayerSuperconductor(BilayerSuperconductivity,
                                  SparseBilayerHamiltonian):
    def __init__(self, H_0:ZKMBSuperconductor, H_1:ZKMBSuperconductor, w_1:float):
        BilayerSuperconductivity.__init__(self, w_1)
        self.H_0 = H_0
        self.H_1 = H_1
        SparseBilayerHamiltonian.__init__(self, self.H_0, self.H_1,
                                          self._get_hopping_z())
        
class SparseBilayerSuperconductorPeriodicInY(BilayerSuperconductivity,
                                             SparsePeriodicBilayerHamiltonianInY):
    def __init__(self,  H_0:ZKMBSuperconductor, H_1:ZKMBSuperconductor, w_1:float):
        BilayerSuperconductivity.__init__(self, w_1)
        self.H_0 = H_0
        self.H_1 = H_1
        SparsePeriodicBilayerHamiltonianInY.__init__(self, self.H_0, self.H_1,
                                          self._get_hopping_z())    
        
class BilayerSuperconductorKY(BilayerSuperconductivity, BilayerHamiltonian):
    r"""ZKM superconductor for a given k in the y direction and magnetic field.
    
    .. math::

        H_{ZKMB} = \frac{1}{2}\sum_k H_{ZKMB,k}
        
        H_{ZKMB,k} = \sum_n^L \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0 + \left(\Delta_0 +2\Delta_1\cos(k) \right) \tau_x\sigma_0
            -2\lambda sin(k) \tau_z\sigma_x
            -\tau_0(B_x\sigma_x+B_y\sigma_y+B_z\sigma_z)
            \right]\vec{c}_n +
            \sum_n^{L-1}\left(\vec{c}^\dagger_n(-t\tau_z\sigma_0 
            -i\lambda \tau_z\sigma_y
            +\Delta_1\tau_x\sigma_0)\vec{c}_{n+1}
            + H.c. \right)
        
        \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},
                   -c^\dagger_{-k,\uparrow})^T
    
        \xi_k = -2tcos(k) - \mu
    """
    def __init__(self, H_0:ZKMBSuperconductorKY, H_1:ZKMBSuperconductorKY, w_1:float):
        BilayerSuperconductivity.__init__(self, w_1)
        self.H_0 = H_0
        self.H_1 = H_1
        BilayerHamiltonian.__init__(self,self.H_0, self.H_1,
                                          self._get_hopping_z())
    