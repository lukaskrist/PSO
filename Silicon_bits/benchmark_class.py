import numpy as np
import scipy.linalg as la
#import spin_chain

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



class DoubleDot:
    def __init__(self, N, T, L, noise = 0.0,epsilon = 1):
        # L is the number of spins
        self.J = -1.0 # nearest interaction
        self.psi0 = 1/np.sqrt(2) *np.array([0,1,0,1,0,0],dtype=complex) 
        psi0 = self.psi0
        length = len(psi0)
        #if gate == 'CZ':  #define gates or rather the intended outcome of the gate
        psi_t = psi0*np.identity(length)
        CZ = np.zeros((length,length))
        CZ[0,0] = 1
        CZ[1,1] = 1
        CZ[2,2] = 1
        CZ[3,3] = -1
        self.psi_target = np.matmul(psi_t,CZ)
        
        
        self.H = np.zeros((6,6),dtype=complex)
        
        E1 = 18.4*1e9
        E2 = 19.7*1e9
        self.N = N
        self.T = T
        #self.L = L # number of spins
        self.noise = noise
        
        self.dt = T/N
        self.tc = 210*1e6
        #self.Hc = -np.copy(self.sp.X)
        self.deltav = (E2-E1)/2
        self.beta = (E2+E1)/2
        self.U1 = 3.5
        self.U2 = 3.5
        self.epsilon = epsilon
        omega = 1
        omega1 = 1.2
        omega2 = 1.3
        self.do1 = omega1-omega
        self.do2 = omega2-omega
        self.step = 0
        self.reset()
    
    def UpdateSingleH(self,H,k,B,phi): 
        '''
        Define hamiltonian depending on the B-field applied.
        Uses hamiltonian H_MW for single qubit gates, needs to be done for each system
        Hamiltonian described in simulation of two electron spins in a double quantum dot
        Has to be done k times for k different signals on the system.
        '''
        H = self.H
        for i in range(k):
            O = B[k]*np.exp(1j*phi[k])                                          #Omega
            O_c = np.conj(O)                                                    #Omega conjugated
            H[0,1] = O_c* np.exp(-1j*self.do1)
            H[0,2] = O_c* np.exp(-1j*self.do2)
            
            H[3,1] = O* np.exp(1j*self.do2)
            H[3,2] = O* np.exp(1j*self.do1)
            
            H[1,0] = O* np.exp(1j*self.do1)
            H[2,0] = O* np.exp(1j*self.do2)
            
            H[1,3] = O_c* np.exp(-1j*self.do2)
            H[2,3] = O_c* np.exp(-1j*self.do1)
        return H
    
    def UpdateDoubleH(self,action): 
        '''
        Define hamiltonian depending on time used in system. 
        Uses hamiltonian H, using the double dot system for multiple qubits
        '''
        t = self.tc
        ex = 1j*self.deltav*t
        
        H = self.H
        
        H[1,4:5] = t*np.exp(-ex)
        H[2,4:5] = -t*np.exp(ex)
        
        H[4:5,1] = t*np.exp(ex)
        H[4:5,2] = -t*np.exp(-ex)
        
        H[4,4] = self.U1+action
        H[5,5] = self.U2-action
        return H
    
    def reset(self):
        self.psi = np.copy(self.psi0)
        self.step = 0
        self.t = 0
        self.H0 = np.zeros((6,6), dtype = complex)

    def observe(self):
        state = np.array([np.real(self.psi), np.imag(self.psi)])
        return np.reshape(state, (state.shape[0]*state.shape[1],))
    
    def update(self, action):
        self.step += 1
        t = self.dt * self.step
        H = self.UpdateDoubleH(action)
        U = la.expm(-1j*H*self.dt)
        
        self.psi = np.dot(U, self.psi)
        state = self.observe()
        #print(self.psi,self.psi_target)
        if self.step == self.N:
            reward = np.abs(np.vdot(self.psi_target, np.diag(self.psi)))**2
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
    
class SingleH:
    def __init__(self, N, T, L, noise = 0.0,epsilon = 1):
        # L is the number of spins
        self.psi0 = np.array([0,0,0,1,0,0],dtype=complex) 
        psi0 = self.psi0
        length = len(psi0)
        #if gate == 'CZ':  #define gates or rather the intended outcome of the gate
        psi_t = psi0*np.identity(length)
        CZ = np.zeros((length,length))
        CZ[0,0] = 1
        CZ[1,1] = 1
        CZ[2,2] = 1
        CZ[3,3] = -1
        CNOT = np.zeros((length,length),dtype=complex) #CNOT gate, should not be possible in this version
        CNOT[0,0] = 1
        CNOT[1,1] = 1
        CNOT[2,3] = 1
        CNOT[3,2] = 1
        X = np.zeros((length,length),dtype=complex) #X gate on both qubits
        X[1,2] = 1
        X[0,3] = 1
        X[2,1] = 1
        X[3,0] = 1
        self.psi_target = np.matmul(psi_t,CZ) #currently this is where the gate is defined. 
        
        
        self.H = np.zeros((6,6),dtype=complex)
        
        E1 = 18.4*1e9
        E2 = 19.7*1e9
        self.N = N
        self.T = T
        #self.L = L # number of spins
        self.noise = noise
        
        self.dt = T/N
        self.tc = 210*1e6
        #self.Hc = -np.copy(self.sp.X)
        self.deltav = (E2-E1)/2
        self.beta = (E2+E1)/2
        self.U1 = 3.5
        self.U2 = 3.5
        self.epsilon = epsilon
        omega = 1
        omega1 = 1.2
        omega2 = 1.3
        self.do1 = omega1-omega
        self.do2 = omega2-omega
        self.step = 0
        self.reset()
    
    def UpdateSingleH(self,k,B): 
        '''
        Define hamiltonian depending on the B-field applied.
        Uses hamiltonian H_MW for single qubit gates, needs to be done for each system
        Hamiltonian described in simulation of two electron spins in a double quantum dot
        Has to be done k times for k different signals on the system.
        '''
        H = self.H
        phi = 1
        for i in range(k):
            O = B*np.exp(1j*phi,dtype=complex)                                  #Omega
            O_c = np.conj(O)                                                    #Omega conjugated
            H[0,1] = O_c* np.exp(-1j*self.do1,dtype=complex)
            H[0,2] = O_c* np.exp(-1j*self.do2,dtype=complex)
            
            H[3,1] = O* np.exp(1j*self.do2,dtype=complex)
            H[3,2] = O* np.exp(1j*self.do1,dtype=complex)
            
            H[1,0] = O* np.exp(1j*self.do1,dtype=complex)
            H[2,0] = O* np.exp(1j*self.do2,dtype=complex)
            
            H[1,3] = O_c* np.exp(-1j*self.do2,dtype=complex)
            H[2,3] = O_c* np.exp(-1j*self.do1,dtype=complex)
        return H
    
    
    def reset(self):
        self.psi = np.copy(self.psi0)
        self.step = 0
        self.t = 0
        self.H0 = np.zeros((6,6), dtype = complex)

    def observe(self):
        state = np.array([np.real(self.psi), np.imag(self.psi)])
        return np.reshape(state, (state.shape[0]*state.shape[1],))
    
    def update(self, action):
        self.step += 1
        k = 1
        H = self.UpdateSingleH(k,action)
        U = la.expm(-1j*H*self.dt)
        
        self.psi = np.dot(U, self.psi)
        state = self.observe()
        if self.step == self.N:
            reward = np.abs(np.vdot(self.psi_target, np.diag(self.psi)))**2
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
        