#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 09:18:29 2022

@author: lukas
"""
import matplotlib.pyplot as plt
import numpy as np
import benchmark_class
N = 20
epochs = 20
a = np.zeros(N)
r_max = np.zeros(epochs)
M_max = np.zeros((epochs,N))
t_list = []
for t in range(epochs):
    T = t/10+2
    t_list.append(T)
    sp = benchmark_class.TwoLevel(N,T)
    for i in range(N):
        for j in range(i,N):
            a[0:i] = -1
            a[i:j] = 0
            a[j:100] = 1
                
            r = sp.roll_out(a)
            if r> r_max[t]:
                r_max[t] = r
                M_max[t,:] = a
                
    print(T)
    
def noise_maker(x):
    noise = np.random.normal(-1,1,len(x))
    x = x+noise*0.1
    return x

r_noise = np.zeros(epochs)
def MaxFunc(M):
    maxval = 1
    for i in range(len(M)):
        if M[i]>maxval:
            M[i] = maxval
        if M[i]< -maxval:
            M[i] = -maxval
    return M

for t in range(epochs):
    T = t/10+2
    sp = benchmark_class.TwoLevel(N,T,noise=0.1)
    r_noise[t] = sp.roll_out(MaxFunc(noise_maker(M_max[t,:])))

ARS = np.load('optimal_M_ARS.npz')
ARS = ARS['arr_0']

r_noise_ars = np.zeros(epochs)

for t in range(epochs):
    T = t/10+2
    sp = benchmark_class.TwoLevel(N,T,noise=0.1)
    r_noise_ars[t] = sp.roll_out(noise_maker(ARS[t,:]))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(t_list,1-r_max,'g+',label = "Infidelity optimal bang bang")
ax1.plot(t_list,1-r_noise,'r+',label = "Infidelity noisy optimal bang bang")
ax1.plot(t_list,1-r_noise_ars,'b*',label = "Infidelity ARS")
ax1.set_xlabel('T')
ax1.set_ylabel('Fidelity')
ax1.set_yscale('log')
ax1.set_title('QSL ARS discrete 2-level system')
ax1.legend()
