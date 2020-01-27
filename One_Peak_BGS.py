#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:50:24 2020

@author: cotdr
"""
import numpy as np
from numba import jit

from matplotlib import pyplot as plt

def main():
    
    gain  = range(1,10,2)
    bfs   = range(100, 1000, 100)
    width = range(100, 1000, 100)
    dfd   = range(0, 1000, 10)
    
    gain_array = One_Peak_BGS(gain, bfs, width, dfd)
    
    plt.figure()
    plt.title('test')
    plt.plot(gain_array[0,4,0,:])
    
    return

@jit
def One_Peak_BGS(gain, bfs, width, dfd):
    
    g_res = np.zeros((len(gain),len(bfs),len(width),len(dfd)))
    
    i=0
    for g in gain:
        j=0
        for b in bfs:
            k=0
            for w in width:
                m=0
                for d in dfd:
                    g_res[i, j, k, m] = g/(1 + ((d-b)/(w/2))**2 )
                    m+=1
                k+=1
            j+=1
        i+=1
        
    return g_res

if __name__ == '__main__':
    """
    Hauptprogramm von hier starten, um Spyder-Crash zu vermeiden
    """
    main()
    



