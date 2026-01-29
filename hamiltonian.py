# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 09:43:29 2024

@author: Gabriel
"""
import numpy as np
from pauli_matrices import (tau_0, tau_y, sigma_0, tau_z, sigma_x, sigma_y, tau_x)
import scipy

def get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_s, w_S, mu_s, mu_S, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                    Lambda, w_1, q_B_x, q_B_y, q_x, q_y):
    r""" A semiconductor plane over a superconductor plane. The semiconductor
    has spin-orbit coupling and magnetic field.
    
    .. math::
        H_\mathbf{k} = \frac{1}{2} (H_s + H_S + H_{w_1})
        
        H_s = -2w_s\left(\cos(k_x)\cos(\Phi_x + q_x) + \cos(k_y)\cos(\Phi_y + q_y)\right)
        \tau_z\sigma_0
        - \left(\sin(k_x)\sin(\Phi_x + q_x) + \sin(k_y)\sin(\Phi_y + q_y)\right)
        \tau_0\sigma_0
        -\mu\tau_z\sigma_0
        + 2\lambda\left(\sin(k_x)\cos(\Phi_x + q_x)\tau_z\sigma_y
        + \cos(k_x)\sin(\Phi_x + q_x)\tau_0\sigma_y
        - \sin(k_y)\cos(\Phi_y + q_y)\tau_z\sigma_x
        - \sin(k_y)\sin(\Phi_y + q_y)\tau_0\sigma_x
        - B_x\tau_0\sigma_x - B_y\tau_0\sigma_y \right)
        
        H_S = -2w_S\left(\cos(k_x)\cos(\Phi_x - q_x) + \cos(k_y)\cos(\Phi_y - q_y)\right)
        \tau_z\sigma_0
        - \left(\sin(k_x)\sin(\Phi_x - q_x) + \sin(k_y)\sin(\Phi_y - q_y)\right)
        \tau_0\sigma_0
        -\mu\tau_z\sigma_0
        + \Delta \tau_x\sigma_0
        
        H_{w_1} = -w_1 \alpha_x\tau_z\sigma_0
            
    """
    H_s = (
        -2*w_s*((np.cos(k_x)*np.cos(phi_x+2*q_B_x+q_x) + np.cos(k_y)*np.cos(phi_y+2*q_B_y+q_y))
               * np.kron(tau_z, sigma_0)
               - (np.sin(k_x)*np.sin(phi_x+2*q_B_x+q_x) + np.sin(k_y)*np.sin(phi_y+2*q_B_y+q_y))
               * np.kron(tau_0, sigma_0)) - mu_s * np.kron(tau_z, sigma_0)
        + 2*Lambda*(np.sin(k_x)*np.cos(phi_x+2*q_B_x) * np.kron(tau_z, sigma_y)
                    + np.cos(k_x)*np.sin(phi_x+2*q_B_x) * np.kron(tau_0, sigma_y)
                    - np.sin(k_y)*np.cos(phi_y+2*q_B_x) * np.kron(tau_z, sigma_x)
                    - np.cos(k_y)*np.sin(phi_y+2*q_B_x) * np.kron(tau_0, sigma_x))
        - B_x*np.kron(tau_0, sigma_x) - B_y*np.kron(tau_0, sigma_y)
         + Delta_s*np.kron(tau_x, sigma_0)
            ) * 1/2
    H_S = (
         # -2*w_S*((np.cos(k_x)*np.cos(phi_x-q_x) + np.cos(k_y)*np.cos(phi_y-q_y))
         #        * np.kron(tau_z, sigma_0)
         #        - (np.sin(k_x)*np.sin(phi_x-q_x) + np.sin(k_y)*np.sin(phi_y-q_y))      # added minus sign because of flux plaquette
         #        * np.kron(tau_0, sigma_0)) - mu_S * np.kron(tau_z, sigma_0)
        -2*w_S*((np.cos(k_x)*np.cos(phi_x+q_x) + np.cos(k_y)*np.cos(phi_y+q_y))   #without q in the superconductor
               * np.kron(tau_z, sigma_0)
               - (np.sin(k_x)*np.sin(phi_x+q_x) + np.sin(k_y)*np.sin(phi_y+q_y))      # added minus sign because of flux plaquette
               * np.kron(tau_0, sigma_0)) - mu_S * np.kron(tau_z, sigma_0)
        
        
        # + 2*Lambda*(np.sin(k_x)*np.cos(phi_x) * np.kron(tau_z, sigma_y)
        #             + np.cos(k_x)*np.sin(phi_x) * np.kron(tau_0, sigma_y)
        #             - np.sin(k_y)*np.cos(phi_y) * np.kron(tau_z, sigma_x)
        #             - np.cos(k_y)*np.sin(phi_y) * np.kron(tau_0, sigma_x))
        - B_x_S*np.kron(tau_0, sigma_x) - B_y_S*np.kron(tau_0, sigma_y)
        + Delta_S*np.kron(tau_x, sigma_0)
            ) * 1/2
    H_w_1 = 1/2 * ( -w_1 * np.kron(tau_z, sigma_0) )
    H = np.block([
            [H_s, H_w_1],
            [H_w_1.conj().T, H_S]
        ])
    return H

def get_Hamiltonian_in_polars(k, theta, phi_x, phi_y, w_s, w_S, mu_s, mu_S, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                              Lambda, w_1, q_x, q_y):
    k_x = k * np.cos(theta)
    k_y = k * np.sin(theta)
    H = get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_s, w_S, mu_s, mu_S, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                        Lambda, w_1, q_x, q_y)
    return H

class Hamiltonian(object):
    r"""A class for 2D Bogoliubov-de-Gennes Hamiltonians.

        Parameters
        ----------
        
        L_x : int
            Number of sites in x-direction (horizontal).
        L_y : int
            Number of sites in y-direction (vertical).
        onsite : ndarray
            4x4 matrix representing the onsite term of the Hamiltonian.
        hopping_x : ndarray
            4x4 matrix representing the hopping term in x of the Hamiltonian.
        hopping_y : ndarray
            4x4 matrix representing the hopping term in y of the Hamiltonian.
    
    .. math ::
       \vec{c_{n,m}} = (c_{n,m,\uparrow},
                        c_{n,m,\downarrow},
                        c^\dagger_{n,m,\downarrow},
                        -c^\dagger_{n,m,\uparrow})^T
              
        H = \sum_i^{L_x}\sum_j^{L_y} \mathbf{c}^\dagger_{i,j}\left[ 
                    \text{onsite} \right] \mathbf{c}_{i,j}\nonumber
        				+ 
                    \sum_i^{L_x}\sum_j^{L_y-1}\left[\mathbf{c}^\dagger_{i,j}
                    \left(\text{hopping_y} \right)\mathbf{c}_{i,j+1}
                    + H.c.\right]
                    +\sum_i^{L_x-1}\sum_j^{L_y}\left[\mathbf{c}^\dagger_{i,j}
                     \left(\text{hopping_x} \right)\mathbf{c}_{i+1,j}
                    + H.c.\right]
    """
    def __init__(self, L_x:int, L_y:int, onsite,
                 hopping_x, hopping_y):
        self.L_x = L_x
        self.L_y = L_y
        self.onsite = onsite
        self.hopping_x = hopping_x
        self.hopping_y = hopping_y
        self.matrix = self._get_matrix().toarray()
    def _index(self, i:int , j:int, alpha:int):    
        #protected method, accesible from derived class but not from object
        r"""Return the index of basis vector given the site (i,j)
        and spin index alpha in {0,1,2,3} for i in {1, ..., L_x} and
        j in {1, ..., L_y}. The site (1,1) corresponds to the lower left real
        space position.
         
            Parameters
            ----------
            i : int
                Site index in x direction. 1<=i<=L_x
            j : int
                Positive site index in y direction. 1<=j<=L_y
            alpha : int
                Spin index. 0<=alpha<=3        
        .. math ::
            \text{Basis vector} = 
           (c_{11}, c_{12}, ..., c_{1L_y}, c_{21}, ..., c_{L_xL_y})^T
           
           \text{index}(i,j,\alpha,L_x,L_y) = \alpha + 4\left(L_y(i-1) +
                                              + j-1\right)
           
           \text{real space}
           
           (c_{1L_y} &... c_{L_xL_y})
                            
           (c_{11} &... c_{L_x1})

        """
        if (i>self.L_x or j>self.L_y):
            raise Exception("Site index should not be greater than \
                            samplesize.")
        if (i<1 or j<1):
            raise Exception("Site index should be a positive integer")
        return alpha + 4*( self.L_y*(i-1) + j-1 )
    def _get_matrix(self):
        r"""
        Matrix of the BdG-Hamiltonian.        
        
        Returns
        -------
        M : ndarray
            Matrix of the BdG-Hamiltonian.
        .. math ::
            \text{matrix space}
            
            (c_{11} &... c_{1L_y})
                             
            (c_{L_x1} &... c_{L_xL_y})
        """
        L_x = self.L_x
        L_y = self.L_y
        M = scipy.sparse.lil_matrix((4*L_x*L_y, 4*L_x*L_y), dtype=complex)
        #onsite
        for i in range(1, L_x+1):    
            for j in range(1, L_y+1):
                for alpha in range(4):
                    for beta in range(4):
                        M[self._index(i , j, alpha), self._index(i, j, beta)]\
                            = 1/2*self.onsite[alpha, beta]
                            # factor 1/2 in the diagonal because I multiplicate
                            # with the transpose conjugate matrix
        #hopping_x
        for i in range(1, L_x):
            for j in range(1, L_y+1):    
                for alpha in range(4):
                    for beta in range(4):
                        M[self._index(i, j, alpha), self._index(i+1, j, beta)]\
                        = self.hopping_x[alpha, beta]
        #hopping_y
        for i in range(1, L_x+1):
            for j in range(1, L_y): 
                for alpha in range(4):
                    for beta in range(4):
                        M[self._index(i, j, alpha), self._index(i, j+1, beta)]\
                        = self.hopping_y[alpha, beta]
        return M + M.conj().T
    def is_charge_conjugation(self):
        """
        Check if charge conjugation is present.

        Parameters
        ----------
        H : Hamltonian
            H_BdG Hamiltonian.

        Returns
        -------
        True or false depending if the symmetry is present or not.

        """
        C = np.kron(tau_y, sigma_y)     #charge conjugation operator
        M = np.kron(np.eye(self.L_x*self.L_y), C)      
        return np.all(np.linalg.inv(M) @ self.matrix @ M
                      == -self.matrix.conj())
    def _get_matrix_periodic_in_y(self):     #self is the Hamiltonian class
        """The part of the tight binding matrix which connects the first
        and last site in the y direction."""
        M = scipy.sparse.lil_matrix((4*self.L_x*self.L_y,
                                     4*self.L_x*self.L_y),
                                    dtype=complex)
        #hopping_y
        for i in range(1, self.L_x+1):
            for alpha in range(4):
                for beta in range(4):
                    M[self._index(i, self.L_y, alpha),
                      self._index(i, 1, beta)] =\
                        self.hopping_y[alpha, beta]
        return M + M.conj().T
    def _get_matrix_periodic_in_x(self):     #self is the Hamiltonian class
        """The part of the tight binding matrix which connects the first
        and last site in the x direction."""
        M = scipy.sparse.lil_matrix((4*self.L_x*self.L_y,
                                     4*self.L_x*self.L_y),
                                    dtype=complex)
        #hopping_y
        for j in range(1, self.L_y+1):
            for alpha in range(4):
                for beta in range(4):
                    M[self._index(self.L_x, j, alpha),
                      self._index(1, j, beta)] =\
                        self.hopping_x[alpha, beta]
        return M + M.conj().T
        

class BilayerHamiltonian(Hamiltonian):
    r"""A class for 2D Bogoliubov-de-Gennes Hamiltonians with a bilayer.

        Parameters
        ----------
        
        L_x : int
            Number of sites in x-direction (horizontal).
        L_y : int
            Number of sites in y-direction (vertical).
        onsite : ndarray
            4x4 matrix representing the onsite term of the Hamiltonian.
        hopping_x : ndarray
            4x4 matrix representing the hopping term in x of the Hamiltonian.
        hopping_y : ndarray
            4x4 matrix representing the hopping term in y of the Hamiltonian.
        hopping_z : ndarray
            4x4 matrix representing the hopping term in z of the Hamiltonian.
    
    .. math ::
       \vec{c_{n,m}} = (c_{n,m,\uparrow},
                        c_{n,m,\downarrow},
                        c^\dagger_{n,m,\downarrow},
                        -c^\dagger_{n,m,\uparrow})^T
              
        H = \sum_i^{L_x}\sum_j^{L_y} \mathbf{c}^\dagger_{i,j}\left[ 
                    \text{onsite} \right] \mathbf{c}_{i,j}\nonumber
        				+ 
                    \sum_i^{L_x}\sum_j^{L_y-1}\left[\mathbf{c}^\dagger_{i,j}
                    \left(\text{hopping_y} \right)\mathbf{c}_{i,j+1}
                    + H.c.\right]
                    +\sum_i^{L_x-1}\sum_j^{L_y}\left[\mathbf{c}^\dagger_{i,j}
                     \left(\text{hopping_x} \right)\mathbf{c}_{i+1,j}
                    + H.c.\right]
    """
    def __init__(self, H_0, H_1, hopping_z):
        self.L_x = H_0.L_x
        self.L_y = H_0.L_y
        self.hopping_z = hopping_z
        self.matrix = self._get_matrix().toarray()
    # def _index(self, i:int , j:int, alpha:int, beta:int):    
    #     #protected method, accesible from derived class but not from object
    #     r"""Return the index of basis vector given the site (i,j)
    #     and spin index alpha in {0,1,2,3} and layer index beta in {0, 1}
    #     for i in {1, ..., L_x} and
    #     j in {1, ..., L_y}. The site (1,1) corresponds to the lower left real
    #     space position.
         
    #         Parameters
    #         ----------
    #         i : int
    #             Site index in x direction. 1<=i<=L_x
    #         j : int
    #             Positive site index in y direction. 1<=j<=L_y
    #         alpha : int
    #             Spin index. 0<=alpha<=3     
    #         beta : int
    #             Layer index. 0 or 1.
    #     .. math ::
    #         \text{Basis vector} = 
    #        (c_{11}, c_{12}, ..., c_{1L_y}, c_{21}, ..., c_{L_xL_y})^T
           
    #        \text{index}(i,j,\alpha,L_x,L_y) = \alpha + 4\left(L_y(i-1)
    #                                           + j-1\right)
           
    #        \text{real space}
           
    #        (c_{1L_y} &... c_{L_xL_y})
                            
    #        (c_{11} &... c_{L_x1})

    #     """
    #     return Hamiltonian._index(i, j, alpha) + beta * (4*self.L_x*self.L_y - 1)
    def _get_matrix(self):
        r"""
        Matrix of the BdG-Hamiltonian.        
        
        Returns
        -------
        M : ndarray
            Matrix of the BdG-Hamiltonian.
        .. math ::
            \text{matrix space}
            
            (c_{11} &... c_{1L_y})
                             
            (c_{L_x1} &... c_{L_xL_y})
        """
        Z = scipy.sparse.kron(scipy.sparse.eye(self.L_x*self.L_y, self.L_x*self.L_y), self.hopping_z)
        # Z = scipy.sparse.csr_array(Z)
        M = scipy.sparse.block_array([[self.H_0._get_matrix(), Z],
                                      [Z.conj().T, self.H_1._get_matrix()]
                                      ])    
        return M
    # def is_charge_conjugation(self):
    #     """
    #     Check if charge conjugation is present.

    #     Parameters
    #     ----------
    #     H : Hamltonian
    #         H_BdG Hamiltonian.

    #     Returns
    #     -------
    #     True or false depending if the symmetry is present or not.

    #     """
    #     C = np.kron(np.kron(tau_y, sigma_y), tau_0)     #charge conjugation operator
    #     M = np.kron(np.eye(self.L_x*self.L_y), C)      
    #     return np.all(np.linalg.inv(M) @ self.matrix @ M
    #                   == -self.matrix.conj())
    def _get_matrix_periodic_in_y(self):     #self is the Hamiltonian class
        """The part of the tight binding matrix which connects the first
        and last site in the y direction."""
        Z = scipy.sparse.kron(self.hopping_z, scipy.sparse.eye(self.L_x*self.L_y, self.L_x*self.L_y))
        # Z = scipy.sparse.csr_array(Z)
        M = scipy.sparse.block_array([[self.H_0._get_matrix_periodic_in_y(), Z],
                      [Z.conj().T, self.H_1._get_matrix_periodic_in_y()]
                      ])    
        return M
    def _get_matrix_periodic_in_x(self):     #self is the Hamiltonian class
        """The part of the tight binding matrix which connects the first
        and last site in the x direction."""
        Z = scipy.sparse.kron(self.hopping_z, scipy.sparse.eye(self.L_x*self.L_y, self.L_x*self.L_y))
        Z = scipy.sparse.csr_array(Z)
        M = scipy.sparse.block_array([[self.H_0._get_matrix_periodic_in_x(), Z],
                      [Z.conj().T, self.H_1._get_matrix_periodic_in_x()]
                      ])    
        return M

class PeriodicHamiltonianInY(Hamiltonian):
    def __init__(self, L_x:int, L_y:int, onsite, hopping_x, hopping_y):
        super().__init__(L_x, L_y, onsite, hopping_x, hopping_y)
        self.matrix = super()._get_matrix().toarray()\
                        + super()._get_matrix_periodic_in_y().toarray()

class PeriodicHamiltonianInYandX(Hamiltonian):
    def __init__(self, L_x:int, L_y:int, onsite, hopping_x, hopping_y):
        super().__init__(L_x, L_y, onsite, hopping_x, hopping_y)
        self.matrix = super()._get_matrix().toarray()\
                        + super()._get_matrix_periodic_in_y().toarray()\
                        + super()._get_matrix_periodic_in_x().toarray()

class SparseHamiltonian(Hamiltonian):
    def __init__(self, L_x:int, L_y:int, onsite, hopping_x, hopping_y):
        self.L_x = L_x      #Do not use super().__init__ because it is sparse
        self.L_y = L_y
        self.onsite = onsite
        self.hopping_x = hopping_x
        self.hopping_y = hopping_y
        self.matrix = self._get_matrix()
        
class SparsePeriodicHamiltonianInY(SparseHamiltonian):
    def __init__(self, L_x:int, L_y:int, onsite, hopping_x, hopping_y):
        super().__init__(L_x, L_y, onsite, hopping_x, hopping_y)
        self.matrix = super()._get_matrix() +\
                        super()._get_matrix_periodic_in_y()



class PeriodicBilayerHamiltonianInY(BilayerHamiltonian):
    def __init__(self, H_0, H_1, hopping_z):
        super().__init__(H_0, H_1, hopping_z)
        self.matrix = super()._get_matrix().toarray()\
                        + super()._get_matrix_periodic_in_y().toarray()

class PeriodicBilayerHamiltonianInYandX(BilayerHamiltonian):
    def __init__(self, H_0, H_1, hopping_z):
        self.matrix = super()._get_matrix().toarray()\
                        + super()._get_matrix_periodic_in_y().toarray()\
                        + super()._get_matrix_periodic_in_x().toarray()

class SparseBilayerHamiltonian(BilayerHamiltonian):
    def __init__(self, H_0, H_1, hopping_z):
        self.H_0 = H_0      #Do not use super().__init__ because it is sparse
        self.H_1 = H_1
        self.L_x = H_0.L_x
        self.L_y = H_0.L_y
        self.hopping_z = hopping_z
        self.matrix = self._get_matrix()
        
class SparsePeriodicBilayerHamiltonianInY(SparseBilayerHamiltonian):
    def __init__(self, H_0, H_1, hopping_z):
        super().__init__(H_0, H_1, hopping_z)
        self.matrix = super()._get_matrix() +\
                        super()._get_matrix_periodic_in_y()
                        
                        
class TrilayerHamiltonian(Hamiltonian):
    r"""A class for 2D Bogoliubov-de-Gennes Hamiltonians with a bilayer.

        Parameters
        ----------
        
        L_x : int
            Number of sites in x-direction (horizontal).
        L_y : int
            Number of sites in y-direction (vertical).
        onsite : ndarray
            4x4 matrix representing the onsite term of the Hamiltonian.
        hopping_x : ndarray
            4x4 matrix representing the hopping term in x of the Hamiltonian.
        hopping_y : ndarray
            4x4 matrix representing the hopping term in y of the Hamiltonian.
        hopping_z : ndarray
            4x4 matrix representing the hopping term in z of the Hamiltonian.
    
    .. math ::
       \vec{c_{n,m}} = (c_{n,m,\uparrow},
                        c_{n,m,\downarrow},
                        c^\dagger_{n,m,\downarrow},
                        -c^\dagger_{n,m,\uparrow})^T
              
        H = \sum_i^{L_x}\sum_j^{L_y} \mathbf{c}^\dagger_{i,j}\left[ 
                    \text{onsite} \right] \mathbf{c}_{i,j}\nonumber
        				+ 
                    \sum_i^{L_x}\sum_j^{L_y-1}\left[\mathbf{c}^\dagger_{i,j}
                    \left(\text{hopping_y} \right)\mathbf{c}_{i,j+1}
                    + H.c.\right]
                    +\sum_i^{L_x-1}\sum_j^{L_y}\left[\mathbf{c}^\dagger_{i,j}
                     \left(\text{hopping_x} \right)\mathbf{c}_{i+1,j}
                    + H.c.\right]
    """
    def __init__(self, H_0, H_1, H_2, hopping_z_0_1, hopping_z_1_2):
        self.L_x = H_0.L_x
        self.L_y = H_0.L_y
        self.hopping_z_0_1 = hopping_z_0_1
        self.hopping_z_1_2 = hopping_z_1_2
        self.matrix = self._get_matrix().toarray()
    def _index(self, i:int , j:int, alpha:int, beta:int):    
        #protected method, accesible from derived class but not from object
        r"""Return the index of basis vector given the site (i,j)
        and spin index alpha in {0,1,2,3} and layer index beta in {0, 1}
        for i in {1, ..., L_x} and
        j in {1, ..., L_y}. The site (1,1) corresponds to the lower left real
        space position.
         
            Parameters
            ----------
            i : int
                Site index in x direction. 1<=i<=L_x
            j : int
                Positive site index in y direction. 1<=j<=L_y
            alpha : int
                Spin index. 0<=alpha<=3     
            beta : int
                Layer index. 0 or 1.
        .. math ::
            \text{Basis vector} = 
           (c_{11}, c_{12}, ..., c_{1L_y}, c_{21}, ..., c_{L_xL_y})^T
           
           \text{index}(i,j,\alpha,L_x,L_y) = \alpha + 4\left(L_y(i-1)
                                              + j-1\right)
           
           \text{real space}
           
           (c_{1L_y} &... c_{L_xL_y})
                            
           (c_{11} &... c_{L_x1})

        """
        return Hamiltonian._index(i, j, alpha) + beta * (4*self.L_x*self.L_y - 1)
    def _get_matrix(self):
        r"""
        Matrix of the BdG-Hamiltonian.        
        
        Returns
        -------
        M : ndarray
            Matrix of the BdG-Hamiltonian.
        .. math ::
            \text{matrix space}
            
            (c_{11} &... c_{1L_y})
                             
            (c_{L_x1} &... c_{L_xL_y})
        """
        Z_0_1 = scipy.sparse.kron(scipy.sparse.eye(self.L_x*self.L_y, self.L_x*self.L_y), self.hopping_z_0_1)
        Z_1_2 = scipy.sparse.kron(scipy.sparse.eye(self.L_x*self.L_y, self.L_x*self.L_y), self.hopping_z_1_2)
        Zeros = scipy.sparse.csr_matrix((4*self.L_x*self.L_y, 4*self.L_x*self.L_y))
        M = scipy.sparse.block_array([[self.H_0._get_matrix(), Z_0_1, Zeros],
                                      [Z_0_1.conj().T, self.H_1._get_matrix(), Z_1_2],
                                      [Zeros, Z_1_2.conj().T, self.H_2._get_matrix()]]  )
        return M
