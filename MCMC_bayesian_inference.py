# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:24:41 2020

@author: Eric Bataller,Luuk Hesselink, Victoria Libucha, Gabriel Raya
"""

import numpy as np
import matplotlib.pyplot as plt
import random

t = np.array([0,0,0,0,0,1,1,1,1,1,1])
x = np.array([[1,2,3],[1,3,2],[1,3,6],[1,5.5,4.5],[1,5,3],[1,7,4],[1,5,6],[1,8,6],[1,9.5,5],[1,9,7],[1,7,8]])
t = t[:,np.newaxis] #We want t to be (11,1) not (11,)

sigm = lambda x: 1/(1+np.e**(-x))

def costM(w,t,x):
    a = x@w 
    y = sigm(a)
    G = float(-(t.T@np.log(np.maximum(y,0.0000001))+(1-t.T)@np.log(1-np.minimum(y,0.9999999))))
    EW = w.T@w /2
    M = float(G + alpha*EW)
    return M, G

def gradM(w,t,x,alpha):
    a = x@w #
    y = sigm(a)
    e = t - y
    g = - x.T @ e
    gM = (alpha*w) + g
    return gM

alpha = 0.01

w = np.random.rand(3, 1)*2 -1 #3 connections, 1 neuron
wMH = np.copy(w) #For the initialization of Metropolis-Hasting
w0,w1,w2= [],[],[] #We will track how w evolves
G_track, M_track = [], [] #Also G and M

g = gradM(w,t,x,alpha)
M, G = costM(w,t,x)
MMH, GMH = costM(w,t,x) #For the initialization of Metropolis-Hasting
a_count = 0

Tau=100
epsilon = 0.055
iterations = 5000

'''
Metropolis-Hesting
'''
# Definition of the posterior probability distribution
def P(MMH):
    return np.exp(-MMH)

w0MH,w1MH,w2MH= [],[],[] #We will track how w evolves
G_trackMH, M_trackMH = [], [] #Also G and M

a_countMH = 0
cov = np.array([[0.01,0,0], [0,0.001,0],[0,0,0.001]])
wMH = [wMH[0][0],wMH[1][0],wMH[2][0]]
iterationsMH = 40000

for i in range(iterationsMH):
    new_wMH = np.random.multivariate_normal(wMH, cov)
    #new_wMH = new_wMH[:,np.newaxis] #We want w to be (3,1) not (3,)
    new_MMH, new_GMH = costM(new_wMH,t,x) 
    a = (P(new_MMH)/P(MMH))
    if a >= 1:
        a_countMH += 1
        wMH = new_wMH
        MMH = new_MMH
        GMH = new_GMH
    else:
        if np.random.uniform(0, 1) > a:
            a_countMH += 1
            wMH = new_wMH
            MMH = new_MMH
            GMH = new_GMH
    w0MH.append(wMH[0])
    w1MH.append(wMH[1])
    w2MH.append(wMH[2])
    G_trackMH.append(new_GMH)
    M_trackMH.append(new_MMH)

acceptance_rateMH = (a_countMH/iterationsMH)*100
print('acceptance MH:', acceptance_rateMH)


'''
HMC
'''

for i in range(iterations):
    p = np.random.rand(3,1)
    H = p.T @ p /2 + M
    w_new = w
    g_new = g
    
    for tau in range(random.randint(100, 200)):
        p = p - epsilon*(g_new/2) #half-step in p
        w_new= w_new + epsilon*p #step in w
        g_new = gradM(w_new,t,x,alpha) #find new gradient
        p = p - epsilon*g_new #half step in p
    
    
    M_new, G_new = costM(w_new,t,x)[0],costM(w_new,t,x)[1]
    H_new = (p.T@p)/2 + M_new #new value for Hamiltonian
    dH = H_new - H #decide whether to accept
    
    if dH<0:
        g = g_new
        w = w_new
        a_count +=1
    elif np.random.uniform(0, 1) < np.exp(-dH):
        g = g_new
        w = w_new
        a_count +=1
        
    w0.append(w[0])
    w1.append(w[1])
    w2.append(w[2])
    G_track.append(G_new)
    M_track.append(M_new)
  
acceptance_rate = (a_count/iterations)*100
print(acceptance_rate)


fig1, ([ax1, ax2,ax3, ax4],[ax5, ax6, ax7, ax8]) = plt.subplots(nrows=2, ncols=4)

ax1.plot(range(iterations), w2,markersize=0.2, label= 'w2')
ax1.plot(range(iterations), w1,markersize=0.2, label= 'w1')
ax1.plot(range(iterations), w0,markersize=0.2, label = 'w0')
ax1.set_ylabel('w')
ax1.set_xlabel('iterations')
ax1.legend(loc=4)
ax1.grid('true')

ax2.plot(w1[500:], w2[500:], 'o', markersize=1) #set burn period to start counting from 250 sample 
ax2.set_ylabel('w2')
ax2.set_xlabel('w1')
ax2.set_title('HMC (Burn-in:500)')
ax2.grid('true')

ax3.plot(range(iterations), G_track, markersize=0.2) #set burn period to start counting from 250 sample 
ax3.set_ylabel('G(w)')
ax3.set_xlabel('iterations')
ax3.set_title('G(iterations)')
ax3.grid('true')

ax4.plot(range(iterations), M_track, markersize=0.2) #set burn period to start counting from 250 sample 
ax4.set_ylabel('M(w)')
ax4.set_xlabel('iterations')
ax4.set_title('M(iterations)')
ax4.grid('true')

ax5.plot(range(iterationsMH), w2MH,markersize=0.2, label= 'w2')
ax5.plot(range(iterationsMH), w1MH,markersize=0.2, label= 'w1')
ax5.plot(range(iterationsMH), w0MH,markersize=0.2, label = 'w0')
ax5.set_ylabel('w')
ax5.set_xlabel('iterations')
ax5.legend( loc=7)
ax5.grid('true')

ax6.plot(w1MH[10000:], w2MH[10000:], 'o', markersize=1) #set burn period to start counting from 250 sample 
ax6.set_ylabel('w2')
ax6.set_xlabel('w1')
ax6.set_title('MH (Burn-in:10000)')
ax6.grid('true')

ax7.plot(range(iterationsMH), G_trackMH, markersize=0.2) #set burn period to start counting from 250 sample 
ax7.set_ylabel('G(w)')
ax7.set_xlabel('iterations')
ax7.set_title('G(iterations)')
ax7.grid('true')

ax8.plot(range(iterationsMH), M_trackMH, markersize=0.2) #set burn period to start counting from 250 sample 
ax8.set_ylabel('M(w)')
ax8.set_xlabel('iterations')
ax8.set_title('M(iterations)')
ax8.grid('true')

