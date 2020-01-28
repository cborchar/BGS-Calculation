#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:50:24 2020
@author: cborchar
"""
import numpy as np
from numba import jit

from matplotlib import pyplot as plt

def main():
    
    gain  = np.ndarray.tolist(np.linspace(1,2,1))
    bfs   = np.ndarray.flatten(calculate_BFS(np.linspace(0, 40, 10), np.linspace(0, 60, 10)))
    width = np.ndarray.tolist(np.linspace(2, 4, 10))
    dfd   = np.ndarray.tolist(np.linspace(10, 11, 1000))
    noise = 0.01
    
    gain_array = One_Peak_BGS(gain, bfs, width, dfd, noise)
    
    plt.figure()
    plt.title('test')
    plt.plot(gain_array[0,0,0,:])
    
    return gain_array, noise, [gain, np.ndarray.flatten(bfs), width, dfd]

@jit
def One_Peak_BGS(gain, bfs, width, dfd, noise_factor):
    
    g_res = np.zeros((len(gain),len(bfs),len(width),len(dfd)))
    
    i=0
    for g in gain:
        j=0
        for b in bfs:
            k=0
            for w in width:
                m=0
                for d in dfd:
                    g_res[i, j, k, m] = g/(1 + ((d-b)/(w/2))**2 ) + np.float32(np.random.normal(0,noise_factor))*g
                    m+=1
                k+=1
            j+=1
        i+=1
        
    return g_res

def calculate_BFS(temperature, strain):
    BFS_res = np.zeros((len(temperature), len(strain)))
    
    i = 0
    for t in temperature:
        j = 0
        for s in strain:
           BFS_res[i][j] = 0.0010546 * (t - 22.0) + 0.0000385 * (s - 0.0) + 10.85 
           j+=1
        i+=1
    return BFS_res


if __name__ == '__main__':
    main()
    
