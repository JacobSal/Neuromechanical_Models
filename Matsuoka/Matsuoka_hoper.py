# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 11:48:58 2021

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

    initial_cond_mats = np.array([np.pi/7,np.pi/14,0,0])
    initial_cond_spring = [.2,0]
    iterable = np.arange(0,30,.5)
    coeff_name = 'B'
    fname = 'B vs frequency'
    avgfrq = []
    ampstore = []
    
    maxpeaks = 5 #max number of peaks to keep in frequency output
    ampstore = []
    frqstore = []
    #Loop
    for j in iterable:
        coeff = [j,2.5,35,35,17.5,17.5*3,6] #bc, wc, h1, h2, t1, t2, c1
        coeffspring = [20000,63.5,0,.17,1] # kc, mc, ac, dc, spring_gain
        mech = mm.Muscle_Mech(1.65)
        mats = mm.Matsuoka(initial_cond_mats,coeff,initial_cond_spring[0])
        springmass = mm.SpringMass(mech.t_s,initial_cond_spring,coeffspring)    
        mech.hoping_model(mats,springmass,20,initial_cond_mats,coeff,False)
        plt.close('all')
        X = np.stack(mats.storeX)
        frqstore.append([mech._avg_freq_profile(X[:,0],maxpeaks),mech._avg_freq_profile(X[:,1],maxpeaks)])
        ampstore.append([max(X[:,0])-min(X[:,0]),max(X[:,1])-min(X[:,1])])
            
    fig = plt.figure('frequency vs {}'.format(coeff_name))
    plt.subplot(2,1,1)
    plt.plot(iterable,np.stack(frqstore)[:,0],label='neuron 1')
    plt.plot(iterable,np.stack(frqstore)[:,1],label='neuron 2')
    plt.xlabel('{}'.format(coeff_name))
    plt.ylabel('frequency (Hz)')
    plt.grid(linestyle='--',linewidth='1')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(iterable,np.stack(ampstore)[:,0],label='neuron 1')
    plt.plot(iterable,np.stack(ampstore)[:,1],label='neuron 2')
    plt.xlabel('{}'.format(coeff_name))
    plt.ylabel('wave amplitude')
    plt.legend()
    plt.grid(linestyle='--',linewidth='1')
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(save_bin,'{}.jpg'.format(fname)),dpi=200,bbox_inches='tight')