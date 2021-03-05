# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 20:41:13 2021

@author: jsalm
"""

import Matsuoka_muscle as mm
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
    plt.close('all')
    coeff = [10,40,30,30,30,30*3,2] #bc, wc, h1, h2, t1, t2, c1
    initial_cond_mats = np.array([np.pi/7,np.pi/14,0,0])
    initial_cond_pend = [np.pi/8,0]
    iterable = np.arange(.2,4,.4)
    coeff_name = 'Muscle Gain'
    fname = 'Muscle Gain vs frequency'
    avgfrq = []
    ampstore = []
    for j in iterable:
        mech = mm.Muscle_Mech(j)
        mats = mm.Matsuoka(initial_cond_mats,coeff)
        pend = mm.Pendulum(mech.t_s,initial_cond_pend)    
        mech.muscle_model(mats,pend,20,initial_cond_mats,coeff)
        plt.close('all')
        X = np.stack(pend.storeX)
        avgfrq.append([mech._avg_freq_profile(X[:,0],5),mech._avg_freq_profile(X[:,1],5)])
        ampstore.append([max(X[:,0])-min(X[:,0]),max(X[:,1])-min(X[:,1])])
        
    fig = plt.figure('frequency vs {}'.format(coeff_name))
    plt.subplot(2,1,1)
    plt.plot(iterable,np.stack(avgfrq)[:,0],label='angular vel')
    plt.plot(iterable,np.stack(avgfrq)[:,1],label='pendulum ang')
    plt.xlabel('{}'.format(coeff_name))
    plt.ylabel('frequency (Hz)')
    plt.grid(linestyle='--',linewidth='1')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(iterable,np.stack(ampstore)[:,0],label='angular vel')
    plt.plot(iterable,np.stack(ampstore)[:,1],label='pendulum ang')
    plt.xlabel('{}'.format(coeff_name))
    plt.ylabel('wave amplitude')
    plt.legend()
    plt.grid(linestyle='--',linewidth='1')
    plt.tight_layout()
    plt.show()
    fig.savefig(r'C:\Users\jsalm\Documents\UF\PhD\Spring 2021\BME6938-Neuromechanics\save_bin\{}.jpg'.format(fname),dpi=200,bbox_inches='tight')