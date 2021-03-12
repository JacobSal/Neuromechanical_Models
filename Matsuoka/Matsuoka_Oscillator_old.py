import sys, os
sys.path.append(r'C:\Users\jsalm\Documents\UF\PhD\Spring 2021\BME6938-Neuromechanics\Berkely Modanna\Py Mimicks')

import numpy as np
import scipy as sp
import pylab as plt
from scipy.fftpack import fft,fftfreq
from scipy.integrate import odeint, RK45


        
class Matsuoka():    
    def __init__(self,initial_cond,coeff):
        """Full Matsuoka Neural Oscillator"""
        self.bc = float(2.5) #coefficient adjusts time course of the adaptation (b=0, 0<b<inf, and b=inf)
        self.wc = float(2.5) #strength of the inhibitory connection bewteen the neurons
        #self.oc = float(np.pi/8) #threshold value below which the neuron does not fire
        self.h1 = float(30) #weight of synaptic conjunction 1 (>0 for excitatory, <0 for inhibitory)
        self.h2 = float(30) #weight of synaptic conjunction 2 (>0 for excitatory, <0 for inhibitory)
        self.t1 = float(55) #tau: a time constant
        self.t2 = float(55*3) #bigT: determine time course of adaptation 
        self.c1 = 2
        self.condition = True
        self.prev_cond = initial_cond
        self.storeX = [initial_cond]
        
    def check_condition(self):
        return (self.t2-self.t1)**2 <= 4*self.t2*self.t1*self.bc
    
    def Stim(self,func,t):
        i = np.where(self.t >= t)[0][0]        
        func = func[int(i)-1]
        return func
    
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
        
    def solve(self,t):        
        X = odeint(self.dALLdt, self.prev_cond, t, tcrit = t, args = (self,oc))

        self.prev_cond = [X[-1,:]]
        self.storeX.append([X[-1,:]])
        # self.storeX = X
        return X
        
    def plot_store(self,n_name):
        """
        Main demo for the Hodgkin Huxley neuron model
        """
        X = np.array(self.storeX)
        x1 = X[:,0]
        x2 = X[:,1]
        v1 = X[:,2]
        v2 = X[:,3]
        docdt = X[:,4]
        oc = X[:,5]
        
        fig = plt.figure("Action Potential Graph of {}".format(n_name),figsize=(20,10))
        plt.subplot(3,1,1)
        plt.title('Matsuoka Plot')
        plt.plot(self.t_ms, x1, 'k')
        plt.plot(self.t_ms, x2, 'k') 
        plt.ylabel('V (mV)')
        plt.xlabel('time (ms)')
        plt.subplot(3,1,2)
        plt.plot(self.t_ms, v1, 'c', label='adaptation n1')
        plt.plot(self.t_ms, v2, 'y', label='adaptation n2')
        plt.ylabel('adaption rate')
        plt.xlabel('time (ms)')
        plt.legend()

        plt.subplot(3,1,3)
        plt.plot(self.t_ms, docdt, 'b--', label = r'$\frac{d\theta_1}{dt}=\theta2$')
        plt.plot(self.t_ms, oc, 'r--', label = r'$\frac{d\theta_2}{dt}=-\frac{b}{m}\theta_2-\frac{g}{L}sin\theta_1$')
        plt.xlabel('time (ms)')
        plt.ylabel('I (mA)')
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
        plt.close()
        return X
    
    def solve(self,t):
        Xp = odeint(self.dALLdt, self.prev_cond, t, tcrit = t, args=(self,))
        self.prev_cond = [Xp[-1,:]]
        self.storeX.append([Xp[-1,:]])
        return Xp

class DifeqInt:
    Ttot = 2 # total time in seconds
    f_s = 1000 # sample frequency (samples/s)
    t_s = np.arange(0,Ttot,1/f_s) #time in s
    t_ms = np.arange(0,Ttot*f_s,1) #time in ms
    
    def __init__(self,funcs,delay):
        self.funcs = funcs #set this as an ordered list
        self.delay = delay #amount of delay in samples 

    def funcs_solver(self):
        delay_ms = 1 #delay in ms
        time_delay = int(self.f_s/1000*delay_ms) #delay in samples based on sample rate
        
        print('starting diffeq solver...')
        for i in range(0,len(self.t_ms)-1):
            t_ms = [self.t_ms[i],self.t_ms[i+1]]
            t_s = [self.t_s[i],self.t_s[i+1]]
            for func in self.funcs:
                
                
            if i > time_delay:
    
    

        
    
if __name__ == "__main__":
    #website: https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/MuscleModeling.ipynb
    initial_cond = [np.pi/7,np.pi/14,0,0]
    mod = Matsuoka(initial_cond)
    X = mod.solve()
    mod.plot_store('matsuoka model')
    
    