import sys, os
sys.path.append(r'C:\Users\jsalm\Documents\UF\PhD\Spring 2021\BME6938-Neuromechanics\Berkely Modanna\Py Mimicks')

import numpy as np
import scipy as sp
import pylab as plt
import os
from scipy.fftpack import fft,fftfreq
from scipy.integrate import odeint, RK45
from scipy.signal import find_peaks

dirname = os.path.dirname(__file__)
save_bin = os.path.join(dirname,"save_bin")
        
class Matsuoka():    
    def __init__(self,initial_cond,coeff,ioc):
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
        self.storeoc = [ioc]
    
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
        
    def plot_store(self,t_ms,n_name,act_fname):
        """
        Main demo for the Hodgkin Huxley neuron model
        act_fname : TYPE string
            DESCRIPTION name of the the activaiton function you are using for the
            matsuoka oscillator (e.g. "Pendulum angle")
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
        plt.ylabel('{}'.format(act_fname))
        plt.legend()
        plt.grid(linestyle='--',linewidth='1')
        plt.tight_layout()
        fig.savefig(os.path.join(save_bin,'{}.png'.format(n_name)),dpi=200,bbox_inches='tight')
        
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
        fig.savefig(os.path.join(save_bin,'pendulum_graph.png'),dpi=200,bbox_inches='tight')
        return X
    
    def solve(self,t):
        Xp = odeint(self.dALLdt, self.prev_cond, t, tcrit = t, args=(self,))
        self.prev_cond = [Xp[-1,0],Xp[-1,1]]
        self.storeX.append([Xp[-1,0],Xp[-1,1]])
        return Xp

class SpringMass():
    #use cham et al 2007
    g = 9.81
    def __init__(self,t,initial_cond,coeff):
        self.kc = float(coeff[0]) # spring stiffness
        self.mc = float(coeff[1]) # body mass
        self.wc = (1/(2*np.pi))*(coeff[0]/coeff[1])**(1/2) # natural frequency
        self.ac = float(coeff[2]) # boundary coefficient
        self.bc = -self.g/(coeff[1])**2 # boundary coefficient
        self.dc = float(coeff[3])
        self.cf = 2*float(coeff[3])*((coeff[0]*coeff[1])**(1/2))
        self.springgain = self.g*coeff[1]*coeff[4] #matsuoka oscillator multiplier
        self.prev_cond = initial_cond
        self.storeX = [initial_cond]
        self.t = t
        
        
    def soft(self,func,t):
        i = np.where(self.t >= t)[0][0]        
        func = func[int(i)-1]
        return func
    
    @staticmethod
    def dALLdt(X,t,self,func):        
        y1, y2 = X #y1 is the velocity, and y2 is the height
        """
        if y1 <= 0:            
            foft = func
            A = np.array([[0,1],[-self.wc**2,-2*self.dc*self.wc]])
            B = np.array([[0],[foft-1]])
        elif y1 > 0:
            A = np.array([[0,1],[0,0]])
            B = np.array([[0],[-1]])
        y = np.array([[y1],[y2]])
        dydt = np.matmul(A,y) + B
        return dydt[0,0], dydt[1,0]
        """
        if y1 <= 0:
            foft = func
            dydt = y2
            dydtdt = foft/self.mc - (self.kc/self.mc)*y1-(self.cf/self.mc)*y2 - self.g
        else:
            dydt = y2
            dydtdt = -self.g
        
        return dydt, dydtdt
        
    def plot_store(self):
        X = np.stack(self.storeX)
        theta_1 = X[:,0]
        theta_2 = X[:,1]
        fig,ax = plt.subplots()
        plt.title('Spring Mass')
        fig.set_size_inches(18.5,10.5)
        ax.plot(self.t,theta_1,color="red",label='height (m)')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('Height (m)',color="red")
        ax2=ax.twinx()
        ax2.plot(self.t,theta_2,label='Length')
        ax2.set_ylabel("Velocity (m/s)",color="blue")
        plt.show()
        fig.savefig(os.path.join(save_bin,'springmass_graph.png'),dpi=200,bbox_inches='tight')
        return X
    
    def solve(self,func,t):
        Xp = odeint(self.dALLdt, self.prev_cond, t, tcrit = t, args=(self,func))
        self.prev_cond = Xp[-1,:]
        self.storeX.append(Xp[-1,:])
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
    Ttot = 3
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
        return y1,y2,y
    
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
       
    def muscle_torque(self,y):
        #for previous implmentation of this in HH model see HodgkinHuxley folder
        torque = y*self.musclegain
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
                pendulum.force = self.muscle_torque(y)     
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
            
    def hoping_model(self,matsuoka,springmass,affgain,n_i,coeff):
        delay_ms = 1 #delay in ms
        time_delay = int(self.f_s/1000*delay_ms) #delay in samples based on sample rate
        print('starting diffeq solver...')
        ub = 0
        bb = 0
        count = 0
        avgfrq = []
        springforce = 0
        for i in range(0,len(self.t_ms)-1):
            #Initia time and Pendulum diffeq
            t_ms = [self.t_ms[i],self.t_ms[i+1]]
            t_s = [self.t_s[i],self.t_s[i+1]]       
            Xp = springmass.solve(springforce,t_s)
            ub += 1
            #delay iteration of force from muscle and activation of neurons
            if i > time_delay:
                mats_X = matsuoka.solve(t_ms,Xp[-1,0])
                y1,_,_ = self.stim(mats_X[-1,0],mats_X[-1,1])
                springforce = self.muscle_torque(y1)
            else:
                matsuoka.storeX.append(n_i)
                matsuoka.storeoc.append(Xp[-1,0])
                springforce = 0
                self.storeT.append(0)
            # judging steady states using average frequency of top 5 peaks
            # if ub > 500:
            #     bb += 1 
            #     avgfrq.append([self._avg_freq_profile(np.stack(matsuoka.storeX)[bb:ub,0],5),
            #                    self._avg_freq_profile(np.stack(matsuoka.storeX)[bb:ub,1],5)])

        Xp = springmass.plot_store()
        matsuoka.plot_store(self.t_ms,'coeff_{}'.format(coeff),"Mass Height (m)")
        self.plot_torque_spring(springmass)
        plt.waitforbuttonpress()
        return 0
    
    def plot_torque_spring(self,spring):
        torque = np.stack(self.storeT)
        length = np.stack(spring.storeX)[:,0]
        fig,ax = plt.subplots()
        plt.title('plot_torque_length')
        fig.set_size_inches(18.5,10.5)
        ax.plot(self.t_ms,torque,color="red",alpha = 0.5,label='torque')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('Force (N)',color="red")
        ax2=ax.twinx()
        ax2.plot(self.t_ms,length,alpha = 0.7, label='Length')
        ax2.set_ylabel("Hopping Height (m)",color="blue")
        plt.show()
        fig.savefig(os.path.join(save_bin,'force_length_graph.png'),dpi=200,bbox_inches='tight')
        
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
        fig.savefig(os.path.join(save_bin,'force_length_graph.png'),dpi=200,bbox_inches='tight')
    
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
        fig.savefig(os.path.join(save_bin,'force_angle_graph.jpg'),dpi=200,bbox_inches='tight')
    


if __name__ == "__main__":
    #website: https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/MuscleModeling.ipynb
    coeff = [2,2.5,35,35,17.5,17.5*3,6] #bc, wc, h1, h2, t1, t2, c1
    coeffspring = [20000,63.5,0,.17,1] # kc, mc, ac, dc, spring_gain
    initial_cond_mats = np.array([np.pi/7,np.pi/14,0,0])
    initial_cond_spring = [.2,0]
    mgain = 800
    
    mech = Muscle_Mech(mgain)
    mats = Matsuoka(initial_cond_mats,coeff,initial_cond_spring[0])
    springmass = SpringMass(mech.t_s,initial_cond_spring,coeffspring)    
    mech.hoping_model(mats,springmass,20,initial_cond_mats,coeff)
    plt.close('all')
    

    #X = mats.plot_store(t_ms,'matsuoka model')
    
    