# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 09:34:32 2021

@author: jsalm
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 14:56:52 2021

@author: jsalm
"""
import tpp_class_ball as tpp
import numpy as np
import os
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)
save_bin = os.path.join(dirname,"save_bin")

if __name__ == '__main__':
    plt.close('all')
    ### PARAMS ###
    #Arm
    plt.close('all')
    n = 2
    pos1 = -np.pi/4 #start mgain: -pi/4, vx: -pi/4
    pos2 = np.pi/4 #start mgain: pi/4, vx: pi/4
    vel = 0
    l1 = (13)*0.0254
    l2 = (12+9)*0.0254
    m1 = 2*5.715264/3
    m2 = 5.715264/3
    damp = 10
    #mech
    mgain = 0 #start vx: 0; for fb: 150
    maxiter = 40
    ballcatch = False
    #ball
    e = 1 #coefficient of restitution
    mass = 2
    vx = -1.5 #start mgain: -1.5
    vy = 0
    y = 1
    x = 2 #start mgain: 2, vx:2,-2 depending on direction of vel, 
    theta = 0
    ### Time ###
    
    Ttot = 3.5 # total time in second
    f_s = 500 # sample frequency (samples/s)
    t_s = np.arange(0,Ttot,1/f_s)
    t_ms = np.arange(0,Ttot*f_s,1)
    
    initial_cond_tpp = [[pos1,pos2],[vel,vel]]
    coeff_tpp = [[l1,l2],[m1,m2],damp]
    initial_cond_ball = [vx,vy,x,y]
    
    ### INIT ###
    Armobj = tpp.TpPendulum(n,initial_cond_tpp,coeff_tpp,mgain,t_s,f_s)
    mech = tpp.Muscle_Mech(mgain,f_s,t_s)
    Ballobj = tpp.Ball(t_s,e,mass,initial_cond_ball)
    
    #### Feedforward ####    
    # f1adj = 0
    # f2adj = 0
    # count = 0
    # obstacle = [(0,0)]
    # while count < maxiter and not ballcatch:
    #     Ballobj = tpp.Ball(t_s,e,mass,initial_cond_ball)
    #     Armobj = tpp.TpPendulum(n,initial_cond_tpp,coeff_tpp,mgain,t_s,f_s)
    #     f1adj,f2adj,ballcatch = mech.ff_iterator(Armobj, Ballobj, maxiter, vx,vy, x, y,f1adj,f2adj, ballcatch)
    #     count += 1
    
    #### Feedback ####
    # xA,yA = Armobj.get_xy()
    # obstacle = (xA[0,2],yA[0,2])
    # for i in range(0,len(t_s)-1):
    #     t = [t_s[i],t_s[i+1]]
    #     dt = abs(t_s[i+1] - t_s[i])        
    #     vx,vy,x,y,ballcatch = Ballobj.simple_ball(vx,vy,x,y,dt,obstacle)
    #     Xp = Armobj.fbsolve(t,x,y,ballcatch)
    #     xA,yA = Armobj.get_xy()
    #     obstacle = (xA[0,2],yA[0,2])
    
    ### Iteration Params ###
    num = -3
    lab = {}
    count = 0
    iterable = np.arange(num,abs(num)+(abs(num)-num)/10,(abs(num)-num)/10)
    coeff_name = 'mgain'
    fname = 'mgain vs frequency'
    catchcoord = []
    finalf1 = []
    finalf2 = []
    #Loop
    for j in iterable:
        if j > 0:
            x = -2
        else:
            x = 2
        #ITERATE PARAMS: mgain, vx, pos2
        initial_cond_tpp = [[pos1,pos2],[vel,vel]]
        lab[count] = coeff_name+": "+("%.2f"%j)
        #### PASTE FEEDForward HERE ####
        f1adj = 0
        f2adj = 0
        iteri = 0
        ballcatch = False
        obstacle = [(0,0)]
        while iteri < maxiter and not ballcatch:
            Ballobj = tpp.Ball(t_s,e,mass,initial_cond_ball)
            Armobj = tpp.TpPendulum(n,initial_cond_tpp,coeff_tpp,mgain,t_s,f_s)
            f1adj,f2adj,ballcatch = mech.ff_iterator(Armobj, Ballobj, maxiter, j,vy, x, y,f1adj,f2adj, ballcatch)
            iteri += 1
            if not iteri < maxiter:
                print('max iteration hit for: ' + lab[count])
                lab[count] = coeff_name+": "+("%.2f"%j)+" NO"
        catchcoord.append([Ballobj.storeB[-1][2],Ballobj.storeB[-1][3]])
        mech.stack_data(Armobj,Ballobj,ballcatch)
        finalf1.append(j+f1adj)
        finalf2.append(j+f2adj)
        count += 1
        ##### PASTE FEEDback HERE ####
        ### INIT ###
        #ball
        # ballcatch = False
        # e = 1 #coefficient of restitution
        # mass = 2
        # vx = -1.5 #start mgain: -1.5
        # vy = 0
        # y = 1
        # x = 2 #start mgain: 2, vx:2,-2 depending on direction of vel, 
        # #objects
        # Armobj = tpp.TpPendulum(n,initial_cond_tpp,coeff_tpp,j,t_s,f_s)
        # Ballobj = tpp.Ball(t_s,e,mass,initial_cond_ball)   
        # #actual loop
        # xA,yA = Armobj.get_xy()
        # obstacle = (xA[0,2],yA[0,2])
        # for i in range(0,len(t_s)-1):
        #     t = [t_s[i],t_s[i+1]]
        #     dt = abs(t_s[i+1] - t_s[i])        
        #     vx,vy,x,y,ballcatch = Ballobj.simple_ball(vx,vy,x,y,dt,obstacle)
        #     Xp = Armobj.fbsolve(t,x,y,ballcatch)
        #     xA,yA = Armobj.get_xy()
        #     obstacle = (xA[0,2],yA[0,2])
        # if not ballcatch:
        #     print('max iteration hit for: ' + lab[count])
        #     lab[count] = coeff_name+": "+("%.2f"%j)+" NO"
        # catchcoord.append([Ballobj.storeB[-1][2],Ballobj.storeB[-1][3]])
        # mech.stack_data(Armobj,Ballobj,ballcatch)
        # count+=1
        continue
        
    Armobj.plot_pendulum_trace()
    Ballobj.plot_ball_trace()
    mech.plot_iter_traces(lab,catchcoord,coeff_name)
    mech.plot_work_energy(iterable,coeff_name)
    B = np.stack(Ballobj.storeB)
    anim = Armobj.animate_ball_pendulum(B[:,2:])
    print(coeff_name+str(np.mean(np.stack(mech.storeW),axis=0)))
    print(coeff_name+str(np.std(np.stack(mech.storeW),axis=0)))
    print(coeff_name+str(np.mean(np.stack(mech.storeE))/1000))
    print(coeff_name+str(np.std(np.stack(mech.storeE))/1000))    
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