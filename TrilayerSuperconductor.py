#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 10:17:21 2025

@author: gabriel
"""

import numpy as np
from pauli_matrices import tau_x, sigma_x, tau_z, sigma_0, sigma_y,\
                            tau_0, sigma_z
from hamiltonian import TrilayerHamiltonian
from ZKMBsuperconductor import ZKMBSuperconductor, ZKMBSuperconductorKY

class TrilayerSuperconductivity():
    def __init__(self, w_0_1, w_1_2):
        self.w_0_1 = w_0_1
        self.w_1_2 = w_1_2
    def _get_hopping_z_0_1(self):
        return -1/2*(self.w_0_1*np.kron(tau_z, sigma_0))
    def _get_hopping_z_1_2(self):
        return -1/2*(self.w_1_2*np.kron(tau_z, sigma_0))

class TrilayerSuperconductor(TrilayerSuperconductivity,
                             TrilayerHamiltonian):
    def __init__(self, H_0:ZKMBSuperconductor, H_1:ZKMBSuperconductor,
                 H_2:ZKMBSuperconductor,
                 w_0_1:float, w_1_2:float):
        TrilayerSuperconductivity.__init__(self, w_0_1, w_1_2)
        self.H_0 = H_0
        self.H_1 = H_1
        self.H_2 = H_2
        TrilayerHamiltonian.__init__(self, self.H_0, self.H_1, self.H_2,
                                     self._get_hopping_z_0_1(),
                                     self._get_hopping_z_1_2())