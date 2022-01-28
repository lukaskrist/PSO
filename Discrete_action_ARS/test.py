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
# Here is the code:
A = ARSTrainer()

#The QSL should be around 3.3 for spin chain and 2.4 for two-level.
T = 0

Noise = 0.01
N = 100
v = 0.4
alpha = 0.4
maxepochs = 40
max_ite = 40
# Make starting pulse u0

pulse = np.random.rand(N)
f_story_B = np.zeros((maxepochs,max_ite))
times_B = np.zeros((maxepochs,max_ite))
m = 0
T_list = []
M_list2 = np.zeros((20,max_ite))

for i in range(max_ite):
    T += 0.1
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
ax1.plot(T_list,np.max(f_story_B,axis = 0),'g+',label = "Fidelity max Waveless")
ax1.set_xlabel('T')
ax1.set_ylabel('Fidelity')
ax1.set_title('QSL ARS discrete 2-level system')
ax1.legend()


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(T_list,1-np.max(f_story_B,axis = 0),'g+',label = "Fidelity max Waveless")
ax1.set_xlabel('T')
ax1.set_ylabel('Fidelity')
ax1.set_yscale('log')
ax1.set_title('QSL ARS discrete 2-level system')
ax1.legend()


l = list(range(max_ite))
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(T_list, l, M_list2, 50, cmap='binary')
ax.set_ylabel('Number of value')
ax.set_xlabel('Times')
ax.set_zlabel('Values')
ax.view_init(60, 35)
fig
