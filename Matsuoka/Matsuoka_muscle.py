import sys, os
sys.path.append(r'C:\Users\jsalm\Documents\UF\PhD\Spring 2021\BME6938-Neuromechanics\Berkely Modanna\Py Mimicks')

import numpy as np
import scipy as sp
import pylab as plt
import os
from scipy.fftpack import fft,fftfreq
from scipy.integrate import odeint, RK45
from scipy.signal import find_peaks


        
class Matsuoka():    
    def __init__(self,initial_cond,coeff):
        """Full Matsuoka Neural Oscillator"""
        self.bc = coeff[0] #float(2.5) #coefficient adjusts time course of the adaptation (b=0, 0<b<inf, and b=inf)
        self.wc = coeff[1] #float(2.5) #strength of the inhibitory connection bewteen the neurons
        #self.oc = float(np.pi/8) #threshold value below which the neuron does not fire
        self.h1 = coeff[2] #float(30) #weight of synaptic conjunction 1 (>0 for excitatory, <0 for inhibitory)
        self.h2 = coeff[3] #float(30) #weight of synaptic conjunction 2 (>0 for excitatory, <0 for inhibitory)
        self.t1 = coeff[4] #float(55) #tau: a time constant
        self.t2 = coeff[5] #float(55*3) #bigT: determine time course of adaptation 
        self.c1 = coeff[6] #2
        self.condition = True
        self.prev_cond = initial_cond
        self.storeX = [initial_cond]
        self.storeoc = [0]
    
        #bc, wc, h1, h2, t1, t2, c1
    def check_condition(self):
        return (self.t2-self.t1)**2 <= 4*self.t2*self.t1*self.bc
        
    @staticmethod
    def dALLdt(X,t,self,oc):
        """
        Integrate

        |  :param X:
        |  :param t:
        |  :return: calculate membrane potential & activation variables
        """
        
        x1, x2, v1, v2  = X
        
        y1 = max(x1,0)
        y2 = max(x2,0)
        
        dx1dt = (-x1-self.bc*v1-self.wc*y2-self.h1*max(oc,0)-self.h2*max(oc,0)+self.c1)/self.t1  #membrane potential
        dx2dt = (-x2-self.bc*v2-self.wc*y1-self.h1*-min(oc,0)-self.h2*-min(oc,0)+self.c1)/self.t1 #membrane potential
        dv1dt = (y1-v1)/self.t2 #degree of the adaptation n1
        dv2dt = (y2-v2)/self.t2 #degree of the adaptation n2
        
        
        return dx1dt, dx2dt, dv1dt, dv2dt
        
    def solve(self,t,oc):        
        X = odeint(self.dALLdt, self.prev_cond, t, tcrit = t, args = (self,oc))

        self.prev_cond = X[-1,:]
        self.storeX.append(X[-1,:])
        self.storeoc.append(oc)
        # self.storeX = X
        return X
        
    def plot_store(self,t_ms,n_name):
        """
        Main demo for the Hodgkin Huxley neuron model
        """
        X = np.stack(self.storeX)
        oc = np.stack(self.storeoc)
        x1 = X[:,0]
        x2 = X[:,1]
        v1 = X[:,2]
        v2 = X[:,3]
        
        fig = plt.figure("Action Potential Graph of {}".format(n_name),figsize=(20,10))
        plt.subplot(3,1,1)
        plt.title('Matsuoka Plot')
        plt.plot(t_ms, x1, 'c', label='Neuron 1')
        plt.plot(t_ms, x2, 'y', label='Neuron 2') 
        plt.ylabel('V (mV)')
        plt.xlabel('time (ms)')
        plt.grid(linestyle='--',linewidth='1')
        plt.legend()
        
        plt.subplot(3,1,2)
        plt.plot(t_ms, v1, 'c', label='adaptation n1')
        plt.plot(t_ms, v2, 'y', label='adaptation n2')
        plt.ylabel('adaption rate')
        plt.xlabel('time (ms)')
        plt.grid(linestyle='--',linewidth='1')
        plt.legend()

        plt.subplot(3,1,3)
        plt.plot(t_ms, oc, 'r--', label = 'activation function')
        plt.xlabel('time (ms)')
        plt.ylabel('Pendulum Angle')
        plt.legend()
        plt.grid(linestyle='--',linewidth='1')
        plt.tight_layout()
        fig.savefig(r'C:\Users\jsalm\Documents\UF\PhD\Spring 2021\BME6938-Neuromechanics\save_bin\{}.png'.format(n_name),dpi=200,bbox_inches='tight')
        
        return X
    
class Pendulum():
    def __init__(self,t,initial_cond):
        pendfreq = 0.1 #set this
        radfreq = pendfreq*2*np.pi
        
        self.R = 9.81/1000/(radfreq**2)
        """ length of rod (m)"""
        self.m = 0.1
        """ mass of pendulum"""
        self.g = 9.81
        """ acceleration of gravity in m/s^2"""
        self.inertia = self.m*self.R**2
        
        self.t = t
        
        self.force = 0
        
        self.prev_cond = initial_cond
        self.storeX = [initial_cond]
        
        self.b = .05*2*(pendfreq*2*np.pi)
        """ dampening coefficient """
        
    @staticmethod
    def dALLdt(theta,t,self):
        theta1 = theta[0]
        theta2 = theta[1]
        #first ode, theta2 = angluar velocity
        dtheta1_dt = theta2
        #second ode
        dtheta2_dt = -((self.b/self.m)*theta2)-((self.g/(self.R))*np.sin(theta1))+self.force/self.inertia
        return dtheta1_dt,dtheta2_dt
        
    def plot_store(self):
        X = np.array(self.storeX)
        theta_1 = X[:,0]
        theta_2 = X[:,1]
        fig = plt.figure('pendulum',figsize=(20,10))
        
        plt.plot(self.t,theta_1,'b--',label=r'$\frac{d\theta_1}{dt}=\theta2$')
        plt.plot(self.t,theta_2,'r--',label=r'$\frac{d\theta_2}{dt}=-\frac{b}{m}\theta_2-\frac{g}{L}sin\theta_1$')
        plt.xlabel('time(s)')
        plt.ylabel('plot')
        plt.legend(loc='best')
        fig.savefig(r'C:\Users\jsalm\Documents\UF\PhD\Spring 2021\BME6938-Neuromechanics\save_bin\pendulum_graph.png',dpi=200,bbox_inches='tight')
        return X
    
    def solve(self,t):
        Xp = odeint(self.dALLdt, self.prev_cond, t, tcrit = t, args=(self,))
        self.prev_cond = [Xp[-1,0],Xp[-1,1]]
        self.storeX.append([Xp[-1,0],Xp[-1,1]])
        return Xp

class SpringMass():
    g = 9.81
    def __init__(self,t,initial_cond,coeff):
        self.kc = int(coeff[0]) # spring stiffness
        self.mc = int(coeff[1]) # body mass
        self.wc = int(coeff[2]) # natural frequency
        self.ac = int(coeff[3]) # boundary coefficient
        self.bc = -self.g/(coeff[2])**2 # boundary coefficinet
    
    @staticmethod
    def dALLdt(X,t,self):
        y = self.ac*np.sin(self.wc)+self.bc*np.cos(self.wc)+self.g/(self.wc)**2
        dydtdt = self.g-self.kc/self.mc*y
        return dydtdt, y
        
    def plot_store(self):
        X = np.array(self.storeX)
        theta_1 = X[:,0]
        theta_2 = X[:,1]
        fig = plt.figure('pendulum',figsize=(20,10))
        
        plt.plot(self.t,theta_1,'b--',label=r'$\frac{d\theta_1}{dt}=\theta2$')
        plt.plot(self.t,theta_2,'r--',label=r'$\frac{d\theta_2}{dt}=-\frac{b}{m}\theta_2-\frac{g}{L}sin\theta_1$')
        plt.xlabel('time(s)')
        plt.ylabel('plot')
        plt.legend(loc='best')
        fig.savefig(r'C:\Users\jsalm\Documents\UF\PhD\Spring 2021\BME6938-Neuromechanics\save_bin\pendulum_graph.png',dpi=200,bbox_inches='tight')
        return X
    
    def solve(self,t):
        Xp = odeint(self.dALLdt, self.prev_cond, t, tcrit = t, args=(self,))
        self.prev_cond = [Xp[-1,0],Xp[-1,1]]
        self.storeX.append([Xp[-1,0],Xp[-1,1]])
        return Xp
    

class DomainFinder():
    def __init__(self):
        self.storex = []
        self.storey = []
    
    def get_ranges(self):
        xmax = np.max(self.storex)
        xmin = np.min(self.storex)
        ymax = np.max(self.storey)
        ymin = np.min(self.storey)
        return (xmin,xmax),(ymax,ymin)
        
class Muscle_Mech(DomainFinder):
    Ttot = 20
    # total time in second
    f_s = 1000 # sample frequency (samples/s)
    t_s = np.arange(0,Ttot,1/f_s)
    t_ms = np.arange(0,Ttot*f_s,1)
    
    def __init__(self,musclegain):
        self.storeT = [0]
        self.storeL = [[0,0]]
        self.musclegain = musclegain
        
    def stim(self,x1,x2):        
        y1=max(x1,0)
        y2=max(x2,0)
        y=y1-y2
        return y
    
    def force_length(self,a,c,dist,theta,Fres,pendR):
        Lmax = dist**2+pendR**2
        Mlength1 = 0
        Mlength2 = 0
        #print('angle: '+str(theta))
        if theta >= 0:
            Mlength1 = (dist**2+pendR**2-2*dist*pendR*np.cos(np.pi/2 - theta))/Lmax
            force1 = a*np.exp(-(Mlength1-Lmax)**2/(2*c**2))
            force2 = Fres
        else:
            Mlength2 = (dist**2+pendR**2-2*dist*pendR*np.cos(np.pi/2 + theta))/Lmax
            force1 = Fres
            force2 = a*np.exp(-(Mlength2-Lmax)**2/(2*c**2))

        self.storeL.append([Mlength1,Mlength2])
        return force1,force2
    
    def force_velocity(self,velocity,vmax):
        Fiso = 0.01
        a = .0001
        b = -.5
        invsigmoid = -a/(b+np.exp(-velocity))+Fiso
        #print(invsigmoid)
        return invsigmoid
       
    def muscle_torque(self,y,Xp,R):
        pendang = Xp[1]
        pendv = Xp[0]
        ### Muscle Params ###
        vmax = .4
        musclegain = self.musclegain
        a = 1
        c = 1
        dist = .1
        Fres = 0.0001
        ###
        FL_2, FL_1 = self.force_length(a,c,dist,pendang,Fres,R)
        FV = self.force_velocity(pendv,vmax)
        torque = y*musclegain #(musclegain/1_000_000*abs(FL_2)*FV*y)-(musclegain/1_000_000*abs(FL_1)*FV*y)
        self.storeT.append(torque)
        return torque
    
    def __generate_power_spec(self,X,f_s,startfrq):
        clipidx = int(len(X)/2)  
        freq = fftfreq(max(X.shape))*f_s
        finder = freq - startfrq
        
        startfrq = np.where(abs(finder)<2)[0][0]
        freq = freq[startfrq:clipidx]       
        X_fft = abs(fft(X).real)[startfrq:clipidx]**2
        return freq, X_fft
    
    def __comp_avg_frq(self,Xfft_at_frq,Xfrq):
        sumval = 0
        maxpwr = sum(Xfft_at_frq)
        for i in range(0,len(Xfft_at_frq)):
            sumval += sumval + (Xfft_at_frq[i]/maxpwr)*Xfrq[i]
        return sumval
    
    def _avg_freq_profile(self,X,peak_cap):
        if X.size == 0:
            raise ValueError("X is empty")
            
        freq, X_fft = self.__generate_power_spec(X,self.f_s,1)    
        X_peaks = find_peaks(X_fft)[0][:peak_cap]
        X_frqs = freq[X_peaks]    
        avgfrq = self.__comp_avg_frq(X_fft[X_peaks],X_frqs)
        return avgfrq
    
    def muscle_model(self,matsuoka,pendulum,affgain,n_i,coeff):
        delay_ms = 1 #delay in ms
        time_delay = int(self.f_s/1000*delay_ms) #delay in samples based on sample rate
        print('starting diffeq solver...')
        ub = 0
        bb = 0
        count = 0
        avgfrq = []
        for i in range(0,len(self.t_ms)-1):
            #Initia time and Pendulum diffeq
            t_ms = [self.t_ms[i],self.t_ms[i+1]]
            t_s = [self.t_s[i],self.t_s[i+1]]            
            Xp = pendulum.solve(t_s)   
            ub += 1
            #delay iteration of force from muscle and activation of neurons
            if i > time_delay:
                mats_X = matsuoka.solve(t_ms,Xp[-1,1])
                y = self.stim(mats_X[-1,0],mats_X[-1,1])
                pendulum.force = self.muscle_torque(y,Xp[-1,:],pendulum.R)               
            else:
                matsuoka.storeX.append(n_i)
                matsuoka.storeoc.append(0)
                pendulum.force = 0
                self.storeT.append(0)
                self.storeL.append([0,0])
            #judging steady states using average frequency of top 5 peaks
            # if ub > 500:
            #     bb += 1 
            #     avgfrq1.append(self._avg_freq_profile(np.stack(matsuoka.storeX)[bb:ub,0],5,self.f_s))
            #     avgfrq2.append(self._avg_freq_profile(np.stack(matsuoka.storeX)[bb:ub,1],5,self.f_s))
            #     if abs(avgfrq1[-2]-avgfrq2[-1]) < 0.1 and count == 0:
            #         print("reached steady state")
            

        # Xp = pendulum.plot_store()
        # matsuoka.plot_store(self.t_ms,'coeff_{}'.format(coeff))
        # self.plot_torque()
        # self.plot_pend_torq(Xp)
        # plt.waitforbuttonpress()
        return 0
    
    def plot_torque(self):
        torque = np.stack(self.storeT)
        length = np.stack(self.storeL)[:,0]
        fig,ax = plt.subplots()
        plt.title('plot_torque_length')
        fig.set_size_inches(18.5,10.5)
        ax.plot(self.t_ms,torque,color="red",label='torque')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('Torque (Nm)',color="red")
        ax2=ax.twinx()
        ax2.plot(self.t_ms,length,label='Length')
        ax2.set_ylabel("Muscle Length (m)",color="blue")
        plt.show()
        fig.savefig(r'C:\Users\jsalm\Documents\UF\PhD\Spring 2021\BME6938-Neuromechanics\save_bin\force_length_graph.png',dpi=200,bbox_inches='tight')
    
    def plot_pend_torq(self,Xp):
        torque = self.storeT
        fig,ax = plt.subplots()
        plt.title('plot_torque_pend')
        fig.set_size_inches(18.5,10.5)
        ax.plot(self.t_ms,torque,color="red",label='torque')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('Torque (Nm)',color="red")
        ax2=ax.twinx()
        ax2.plot(self.t_ms,Xp[:,1],label='pendulum angle')
        ax2.set_ylabel("Pendulum Angle (rads)",color="blue")
        plt.show()
        fig.savefig(r'C:\Users\jsalm\Documents\UF\PhD\Spring 2021\BME6938-Neuromechanics\save_bin\force_angle_graph.jpg',dpi=200,bbox_inches='tight')
    


if __name__ == "__main__":
    #website: https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/MuscleModeling.ipynb
    Ttot = 2
    # total time in second
    f_s = 1000 # sample frequency (samples/s)
    t_s = np.arange(0,Ttot,1/f_s)
    t_ms = np.arange(0,Ttot*f_s,1)
    delay_ms = 1
    time_delay = int(f_s/1000*delay_ms)
    oc = 2*np.sin(t_s*(20))
    maxpeaks = 5 #max number of peaks to keep in frequency output
    iterable  = np.arange(5,50,1)
    coeff_name = 't1'
    fname = 't1 vs frequency'
    ampstore = []
    frqstore = []
    
    for j in iterable:
        initial_cond_mats = np.array([np.pi/7,np.pi/14,0,0])
        initial_cond_pend = [np.pi/8,0]
        coeff = [2.5,2.5,30,30,j,55*3,2] #bc, wc, h1, h2, t1, t2, c1
        mats = Matsuoka(initial_cond_mats,coeff)
        pend = Pendulum(t_s,initial_cond_pend)
        
        for i in range(0,len(t_ms)-1):
            t_ms_i = [t_ms[i],t_ms[i+1]]
            t_s_i = [t_s[i],t_s[i+1]]
            X_i = mats.solve(t_ms_i,oc[i])
        
        X = np.stack(mats.storeX)
        freq, X1_fft = generate_power_spec(X[:,0],f_s,1)
        _, X2_fft = generate_power_spec(X[:,1],f_s,1)
        
        X1_peaks = find_peaks(X1_fft)[0][:maxpeaks]
        X2_peaks = find_peaks(X2_fft)[0][:maxpeaks]
        
        X1_frqs = freq[X1_peaks]
        X2_frqs = freq[X2_peaks]
        
        avgfrq1 = comp_avg_frq(X1_fft[X1_peaks],X1_frqs)
        avgfrq2 = comp_avg_frq(X2_fft[X2_peaks],X2_frqs)
        
        frqstore.append([avgfrq1,avgfrq2])
        ampstore.append([max(X[:,0])-min(X[:,0]),max(X[:,0])-min(X[:,0])])
        
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
    fig.savefig(r'C:\Users\jsalm\Documents\UF\PhD\Spring 2021\BME6938-Neuromechanics\save_bin\{}.jpg'.format(fname),dpi=200,bbox_inches='tight')
    

    #X = mats.plot_store(t_ms,'matsuoka model')
    
    