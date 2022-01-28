"""
Created on Thu Feb 25 10:58:00 2021
@author: loklu
"""


import numpy as np
import time
import benchmark_class

def MaxFunc(M):
    maxval = 1
    for i in range(len(M)):
        if M[i]>maxval:
            M[i] = maxval
        if M[i]< -maxval:
            M[i] = -maxval
    return M

def make_continuous(delta,eps):
    r = 0
    for i in range(1,len(delta)):
        if abs(delta[i]-delta[i-1]) < eps:
            r += 1
    return r/(len(delta)*5)


class ARSTrainer():

    def __init__(self):
        #print("ARS-Wave initialised")
        #initialize global parameters
        self.N = 20
        
        
    def train(self,pulse,N,T,alpha,v,maxepochs = 20,data = np.zeros(1),A = np.zeros(1), Noise = None,L = None):
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
        if L != None:
            sp = benchmark_class.SpinChain(N,T,L,Noise)
        else:
            sp = benchmark_class.TwoLevel(N,T,Noise)
        ### assertions
        #assert psi0.size == psi_target.size
        ### initialize
        epoch = 0
        p = 10
        AccHist = []
        M = np.zeros(N)#pulse #np.zeros((l,N))
        ### main loop
        t0 = time.time()
        times = []
        #depends on the data
        F_list = np.zeros(2)
        eps = 0.1
        
        while epoch < maxepochs:
            epoch += 1

            samples = np.random.normal(size = (p,N))
            r_plus_list = []
            r_minus_list = []
            M_update = np.zeros((N))
            for i in range(p):
                delta_plus = M+samples[i,:]*v
                delta_plus = MaxFunc(delta_plus)
                r_plus = make_continuous(delta_plus,eps)
                delta_minus = M-samples[i,:]*v
                delta_minus = MaxFunc(delta_minus)
                r_minus = make_continuous(delta_minus,eps)
                
                r_plus += sp.roll_out(delta_plus)
                r_minus += sp.roll_out(delta_minus)
                #if r_plus > 0.15 or r_minus > 0.15:
                r_plus_list.append(r_plus)
                r_minus_list.append(r_minus)
                M_update += alpha/N *(r_plus-r_minus )*samples[i,:]
                #print(M_update)
                
            # update by using the standard deviation
            std = np.std([r_plus_list,r_minus_list])
            
            if std != 0:
                M_update /= std
            M_old = M
            M += M_update
            M = MaxFunc(M)
            M = make_continuous(M,eps)
            #print(np.sum(M_old-M))
            AccHist.append(np.max([r_plus_list,r_minus_list]))
            #AccHist.append(sp.roll_out(M))
            times.append(time.time()-t0)
            ### END CODE
        
        return M, F_list, AccHist,times
