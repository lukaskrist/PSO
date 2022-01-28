#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 12:00:21 2021

@author: lukas
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 11:12:32 2021

@author: loklu
"""

import numpy as np
from ARS_discrete_a import ARSTrainer
import matplotlib.pyplot as plt
from Discrete_single_qubit import SingleQubitTrainer
# Here is the code:
A = ARSTrainer()
B = SingleQubitTrainer()
#The QSL should be around 3.3 for spin chain and 2.4 for two-level.
T = 0

Noise = 1
N = 100
v = 2
alpha = 2

maxepochs = 10
max_ite = 20
# Make starting pulse u0

pulse = np.random.rand(N)
f_story_A = np.zeros((maxepochs,max_ite))
times_A = np.zeros((maxepochs,max_ite))
M_list = np.zeros((20,max_ite))
f_story_B = np.zeros((maxepochs,max_ite))
times_B = np.zeros((maxepochs,max_ite))
M_list2 = np.zeros((20,max_ite))
T_list = []



m = 0
for i in range(max_ite):
    T += 20
    M_list[:,i],__,f_story_A[:,i],times_A[:,i] = B.train(pulse,N,T,alpha,v,Noise = Noise,maxepochs=maxepochs)
    M_list2[:,i],__,f_story_B[:,i],times_B[:,i] = A.train(pulse,N,T,alpha,v,Noise = Noise,maxepochs=maxepochs)
    pulse = np.random.rand(N)
    T_list.append(T)
    print(i)
    
    
#time_average_A = np.average(times_A,axis = 0)
#f_story_avg_A = np.average(f_story_A,axis = 0)
#f_story_std_A = np.std(f_story_A,axis = 0)
#f_story_max_A = np.max(f_story_A,axis = 0)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(T_list,np.max(f_story_A,axis = 0),'g+',label = "Fidelity max Waveless")
ax1.set_xlabel('T')
ax1.set_ylabel('Fidelity')
ax1.set_title('QSL ARS discrete single gate')
ax1.legend()


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(T_list,1-np.max(f_story_A,axis = 0),'g+',label = "inFidelity max Waveless")
ax1.set_xlabel('T')
ax1.set_ylabel('inFidelity')
ax1.set_yscale('log')
ax1.set_title('QSL ARS discrete single gate infidelity')
ax1.legend()


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(T_list,np.max(f_story_B,axis = 0),'g+',label = "Fidelity max Waveless")
ax1.set_xlabel('T')
ax1.set_ylabel('Fidelity')
ax1.set_title('QSL ARS discrete DoubleDot')
ax1.legend()


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(T_list,1-np.max(f_story_B,axis = 0),'g+',label = "inFidelity max Waveless")
ax1.set_xlabel('T')
ax1.set_ylabel('inFidelity')
ax1.set_yscale('log')
ax1.set_title('QSL ARS discrete DoubleDot infidelity')
ax1.legend()

