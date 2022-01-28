"""
Created on Thu Feb 25 10:58:00 2021
@author: loklu
"""


import numpy as np
import time
import benchmark_class
from scipy import interpolate

def MaxFunc(M):
    maxval = 10000
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

def noise_maker(x):
    noise = np.random.normal(-1,1,len(x))
    x = x+noise*0.01
    return x

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

        sp = benchmark_class.DoubleDot(N,T,L,Noise)
        
        ### initialize
        epoch = 0
        p = 10
        AccHist = []
        M = np.random.rand((20))                                                #pulse #np.zeros((l,N))
        ### main loop
        t0 = time.time()
        times = []
        
        x_new = np.linspace(0,19,N)
        r_max = 0
        F_list = np.zeros(2)
        interpol = np.arange(20)
        while epoch < maxepochs:
            epoch += 1

            samples = np.random.normal(size = (p,20))
            r_plus_list = []
            r_minus_list = []
            M_update = np.zeros((20))

            for i in range(p):
                delta_plus = M+samples[i,:]*v                                  #deltas functioned     
                delta_plus = MaxFunc(delta_plus)
                
                delta_minus = M-samples[i,:]*v
                delta_minus = MaxFunc(delta_minus)
                
                int_plus = interpolate.interp1d(interpol, delta_plus,kind='cubic') #interpolate
                int_minus = interpolate.interp1d(interpol, delta_minus,kind='cubic')
                
                int_plus = int_plus(x_new)                                      #Make int into real numbers
                int_minus = int_minus(x_new)
                
                noised_plus = noise_maker(np.array(int_plus))                   #Make noise on actions
                noised_minus = noise_maker(np.array(int_minus))
                
                r_plus = sp.roll_out(noised_plus)                               #Roll-out
                r_minus = sp.roll_out(noised_minus)
                
                r_plus_list.append(r_plus)
                r_minus_list.append(r_minus)
                
                M_update += alpha/N *(r_plus-r_minus )*samples[i,:]
                
                if r_plus > r_max:                                             #Pocket algorithm
                    r_max = r_plus
                    M_max = delta_plus
                if r_minus > r_max:
                    r_max = r_minus
                    M_max = delta_minus
                
                
            # update by using the standard deviation
            std = np.std([r_plus_list,r_minus_list])
            
            if std != 0:
                M_update /= std
                                                                                #M_old = M
            M += M_update
            M = MaxFunc(M)
            #M = make_continuous(M,eps)
                                                                                #print(np.sum(M_old-M))
            AccHist.append(np.max([r_plus_list,r_minus_list]))
                                                                                #AccHist.append(sp.roll_out(M))
            times.append(time.time()-t0)
            ### END CODE
        
        return M_max, F_list, AccHist,times
