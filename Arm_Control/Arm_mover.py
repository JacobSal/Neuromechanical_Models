# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 14:56:52 2021

@author: jsalm
"""
import tpp_class_V2 as tpp
import numpy as np
import os
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)
save_bin = os.path.join(dirname,"save_bin")

if __name__ == '__main__':
    plt.close('all')
    # linear algebra help: http://homepages.math.uic.edu/~jan/mcs320/mcs320notes/lec37.html
    ### PARAMS ###
    #Arm
    n = 2
    pos1 = np.pi/4
    pos2 = -np.pi/4
    vel = 0
    l1 = (13)*0.0254
    l2 = (12+9)*0.0254
    m1 = 2*5.715264/3
    m2 = 5.715264/3
    damp = 5
    initial_cond_tpp = [[pos1,pos2],[vel,vel]]
    coeff_tpp = [[l1,l2],[m1,m2],damp]
    
    #mech
    mgain = 15
    ang_desired = [np.pi/8,-np.pi/8]
    
    ### Time ###
    Ttot = 5 # total time in second
    f_s = 500 # sample frequency (samples/s)
    t_s = np.arange(0,Ttot,1/f_s)
    t_ms = np.arange(0,Ttot*f_s,1)
    
    ### Iteration Params ###
    num = 1
    iterable = np.arange(num/10,num*10,num)
    coeff_name = 'B'
    fname = 'B vs frequency'
    
    #Loop
    for j in iterable:
        mech = tpp.Muscle_Mech(mgain,f_s,t_s)
        pend = tpp.TpPendulum(n,initial_cond_tpp,coeff_tpp,mgain,t_s,f_s)
        mech.arm_model(pend,ang_desired,plot=True)
        X = np.stack(pend.storeP)[:,:n]
        T = np.stack(mech.storeT)
        break
    anim = pend.animate_pendulum()
    pend.plot_pendulum_trace()
    
    # fig = plt.figure('frequency vs {0}'.format(coeff_name))
    # plt.subplot(2,1,1)
    # plt.plot(iterable,np.stack(frqstore)[:,0],label='neuron 1')
    # plt.xlabel('{0}'.format(coeff_name))
    # plt.ylabel('frequency of pendulum (Hz)')
    # plt.grid(linestyle='--',linewidth='1')
    # plt.legend()
    # plt.subplot(2,1,2)
    # plt.plot(iterable,np.stack(workstore)[:,0],label='neuron 1')
    # plt.xlabel('{0}'.format(coeff_name))
    # plt.ylabel('work (Nm)')
    # plt.legend()
    # plt.grid(linestyle='--',linewidth='1')
    # plt.tight_layout()
    # plt.show()
    # fig.savefig(os.path.join(save_bin,'{0}.jpg'.format(fname)),dpi=200,bbox_inches='tight')