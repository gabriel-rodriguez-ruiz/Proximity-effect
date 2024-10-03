# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:23:29 2024

@author: Gabriel
"""

import numpy as np
import multiprocessing

n_cores = 4


if __name__ == "__main__":
    print(3)
    B_values = np.linspace(0, 3, 10)
    def printing(a):
        return print(a)
    with multiprocessing.Pool(n_cores) as pool:
        results_pooled = pool.map(printing, B_values)