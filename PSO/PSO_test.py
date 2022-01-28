#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 14:15:21 2021

@author: lukas
"""

import numpy as np
#import benchmark_class
import PSO
import matplotlib.pyplot as plt
N = 20
T = 1/3

noise = 0 # remember noise is optional

#two_level = benchmark_class.TwoLevel(N, T, noise)

#alist = 2*(np.random.rand(N)-0.5)

# this is how you evaluate it, with step-per-step updates

#for step in range(0, N):
    
#    _, r = two_level.update(alist[step])
#    print("r:", r)

# this is how you use roll_out

#G = two_level.roll_out(alist)
#print("G:", G)
Tlist = []
# Note, if the noise is different from zero, you get a different result each time. 
#pulse = np.random.rand(N)#np.array([ 1.        ,  0.75186758,  0.18700439, -0.90246136, -1.        ,
       #-1.        , -1.        , -1.        ,  0.37882948,  0.21337853,
       # 0.21337512,  0.37883106, -1.        , -1.        , -1.        ,
       #-1.        , -0.90246192,  0.18700451,  0.75186756,  1.        ]) #np.random.rand
w = 0.4
c1 = 0.1
c2 = 0.3
maxite = 40
maxepochs = 50
f_story = np.zeros((maxepochs,maxite))
times = np.zeros((maxepochs,maxite))
M_list = np.zeros((N,maxite))
PSO = PSO.PSOTrainer()

for i in range(maxite):
    T += 0.1
    f_story[:,i],times[:,i] = PSO.train(N,T,w,c1,c2, Noise = noise,maxepochs=maxepochs)
        
    Tlist.append(T)
    print(T,f_story[-1,i])


time_average_A = np.average(times,axis = 0)
f_story_avg_A = np.average(f_story,axis = 0)
f_story_std_A = np.std(f_story,axis = 0)
f_story_max_A = np.max(f_story,axis = 0)
fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax1.plot(time_average_A,f_story_avg_A,'*',label = "Fidelity max Waveless")
#ax1.plot(times,f_story,'*',label = "Fidelity max Waveless")
ax1.plot(Tlist,f_story_max_A,'*',label = "Fidelity max Waveless")
ax1.set_xlabel('T ')
ax1.set_ylabel('Fidelity')
ax1.set_title('QSL PSO graph-Max')
#ax1.legend()
ax1.set_title('QSL PSO graph')
fig.show()