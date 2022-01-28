import numpy as np
import scipy.linalg as la
import spin_chain

'''
This script contains a series of benchmark class for a RL algorithm.
Each class is constructed with the following methods.

A constructor: A function that initializes the test class with a given set of
                parameters. 

reset: A function that resets the class to it's initial state

observe: A function that returns the current state of the system

update: A function that updates the state via a selected action, and returns a state
        and a reward

roll_out: A function that evaluated an entire episode in one-go based on a list of actions

'''



class TwoLevel:
    
    def __init__(self, N, T, noise = 0.0):
        
        self.psi0 = np.zeros((2), dtype = complex)
        self.psi0[0] = 1.0
        
        self.psi_target = np.zeros((2), dtype = complex)
        self.psi_target[-1] = 1.0
        
        self.sigma_z = np.array([[1,0], [0,-1]], dtype = complex)
        self.sigma_x = np.array([[0,1], [1,0]], dtype = complex)
        
        self.N = N
        self.T = T
        self.noise = noise
        self.dt = T/N
        
        self.reset()
        
    
    def reset(self):
        self.psi = np.copy(self.psi0)
        self.step = 0
        self.H0 = np.random.normal(loc = 1.0, scale = self.noise)*self.sigma_z
        self.Hc = np.random.normal(loc = 1.0, scale = self.noise)*self.sigma_x
        
    def observe(self):
        state = np.array([np.real(self.psi), np.imag(self.psi)])
        return np.reshape(state, (4,))
    
    def update(self, action):
        H = self.H0 + action*self.Hc
        U = la.expm(-1j*H*self.dt)
        
        self.psi = np.dot(U, self.psi)
        
        self.step += 1
        
        state = self.observe()
        
        if self.step == self.N:
            reward = np.abs(np.vdot(self.psi_target, self.psi))**2
        else:
            reward = 0
            
        return state, reward
    
    def roll_out(self, action_list):
        self.reset()
        G = 0
        for step in range(0, self.N):
            
            _, r = self.update(action_list[step])
            G += r
            
        return G
            
        
                
            
class SpinChain:
    def __init__(self, N, T, L, noise = 0.0):
        # L is the number of spins
        self.J = -1.0 # nearest interaction
        
        self.N = N
        self.T = T
        self.L = L # number of spins
        self.noise = noise
        
        self.dt = T/N
        
        self.sp = spin_chain.Spin_chain(L)
        
        self.Hc = -np.copy(self.sp.X)
        
        
        self.psi0 = np.zeros((self.sp.dim), dtype = complex)
        self.psi0[0] = 1.0
        
        self.psi_target = np.zeros((self.sp.dim), dtype = complex)
        self.psi_target[-1] = 1.0
    
        self.step = 0
        self.reset()
    
    def reset(self):
        self.psi = np.copy(self.psi0)
        self.step = 0
        
        self.H0 = np.zeros((self.sp.dim, self.sp.dim), dtype = complex)
        
        for idx in range(0, self.L):
            self.H0 += np.random.normal(loc = self.J, scale = self.noise)*self.sp.get_ZZ(idx, np.mod(idx + 1,self.L))
            
    def observe(self):
        state = np.array([np.real(self.psi), np.imag(self.psi)])
        return np.reshape(state, (state.shape[0]*state.shape[1],))
    
    def update(self, action):
        H = self.H0 + action*self.Hc
        U = la.expm(-1j*H*self.dt)
        
        self.psi = np.dot(U, self.psi)
        
        self.step += 1
        
        state = self.observe()
        
        if self.step == self.N:
            reward = np.abs(np.vdot(self.psi_target, self.psi))**2
        else:
            reward = 0
            
        return state, reward
    
    def roll_out(self, action_list):
        self.reset()
        G = 0
        for step in range(0, self.N):
            
            _, r = self.update(action_list[step])
            G += r
            
        return G    
        