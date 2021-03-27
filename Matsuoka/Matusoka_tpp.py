# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 12:17:21 2021

@author: jsalm
"""


import Matsuoka_muscle as mm
import numpy as np
import os
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)
save_bin = os.path.join(dirname,"save_bin")

if __name__ == '__main__':
    plt.close('all')
    pos = -np.pi/4
    vel = 1
    pendfreq = 0.1 #set this
    radfreq = pendfreq*2*np.pi
    l1 = 9.81/1000/(radfreq**2)
    m1 = .1
    n = 1
    initial_cond_mats = np.array([np.pi/7,np.pi/14,0,0])
    initial_cond_tpp = [[pos],[vel]]
    num = 30
    iterable = np.arange(num/10,num*10,num)
    coeff_name = 'B'
    fname = 'B vs frequency'
    avgfrq = []
    ampstore = []
    
    maxpeaks = 5 #max number of peaks to keep in frequency output
    frqstore = []
    workstore = []
    #Loop
    for j in iterable:
        coeff = [j,27,45,45,65,65*3,6] #bc, wc, h1, h2, t1, t2, c1
        coeff_tpp = [[l1],[m1],.2]
        mgain = .02      
        mech = mm.Muscle_Mech(mgain)
        mats = [mm.Matsuoka(initial_cond_mats,coeff,mm.TpPendulum._conv_ang(initial_cond_tpp[0][i])) for i in range(n)]
        pend = mm.TpPendulum(n,initial_cond_tpp,coeff_tpp,mech.t_s,mech.f_s)
        mech.three_pend_model(mats,pend,initial_cond_mats,initial_cond_tpp,n=1,plot=False)
        plt.close('all')
        # anim = pend.animate_pendulum()
        X = np.stack(pend.storeP)[:,:n]
        frqstore.append([mech._avg_freq_profile(X[:,0],maxpeaks)])
        workstore.append(pend.work)
            
    fig = plt.figure('frequency vs {0}'.format(coeff_name))
    plt.subplot(2,1,1)
    plt.plot(iterable,np.stack(frqstore)[:,0],label='neuron 1')
    plt.xlabel('{0}'.format(coeff_name))
    plt.ylabel('frequency of pendulum (Hz)')
    plt.grid(linestyle='--',linewidth='1')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(iterable,np.stack(workstore)[:,0],label='neuron 1')
    plt.xlabel('{0}'.format(coeff_name))
    plt.ylabel('work (Nm)')
    plt.legend()
    plt.grid(linestyle='--',linewidth='1')
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(save_bin,'{0}.jpg'.format(fname)),dpi=200,bbox_inches='tight')