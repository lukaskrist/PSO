#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 10:42:22 2021

@author: lukas
"""

import numpy as np
import time
import benchmark_class
import random

def MaxFunc(M,p,N):
    maxval = 1
    for i in range(p):
        for k in range(N):
            if M[i,k]>maxval:
                M[i,k] = maxval
            if M[i,k]< -maxval:
                M[i,k] = -maxval
    return M
class PSOTrainer():

    def __init__(self):
        #print("ARS-Wave initialised")
        #initialize global parameters
        self.N = 20
        
        
    def train(self,N,T,w,c1,c2,maxepochs = 20,data = None,A = np.zeros(1), Noise = None,L = None):
        """
        Implement Basic random search
        psi = our start configuration of the wave function, will be updated along the way
        psi_t = the target function (will start with going from 0 state to 1 state, but will )
        u0 = the starting control vector, which will be updated. 
        alpha = step-size 
        N = number of directions sampled per iteration 
        v = standard deviation of the exploration noise
        p = number of horizons
        maxepochs = the maximum amount of epochs that we go through before converging
        theta = the update vector for the u0
        if we have data we put that in the data, but will first be implemented later, for now that is just none
        """
        assert T>0
        if L != None:
            sp = benchmark_class.SpinChain(N,T,L,Noise)
        else:
            sp = benchmark_class.TwoLevel(N,T,Noise)

        ### initialize
        p = 2*N#2*N #Number of particles
        AccHist = []

        ### main loop
        t0 = time.time()
        times = []
        
        # initialize position and velocity (maybe need to change values)
        #x = np.random.uniform(-1,1,size = (p,N))
        #v = np.random.uniform(-0.5,0.5,size = (p,N))
        x = np.random.normal(size = (p,N))
        v = np.random.normal(size = (p,N))
        k = 0
        F = 0 #best valued fidelity
        F_all = np.zeros(p)
        #x_best = np.zeros(N)
        while k < maxepochs:
            k += 1
            delta = np.zeros(p)
            for i in range(p):
                l = sp.roll_out(x[i,:])
                delta[i] = l
                if delta[i] > F:
                    F = l
                    x_best = x[i,:]
                if l > F_all[i]:
                    F_all[i] = l
            for i in range(p):
                for j in range(N):
                    v[i,j] = w*v[i,j]
                    v[i,j] += c1*random.random()*(F_all[i]-x[i,j])
                    v[i,j] += c2*random.random()*(x_best[j]-x[i,j])
                    x[i,j] = x[i,j]+v[i,j]
            x = MaxFunc(x,p,N)

            AccHist.append(F)
            times.append(time.time()-t0)
            
        return AccHist,times