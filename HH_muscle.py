import sys, os
sys.path.append(r'C:\Users\jsalm\Documents\UF\PhD\Spring 2021\BME6938-Neuromechanics\Berkely Modanna\Py Mimicks')

import numpy as np
import scipy as sp
import pylab as plt
from scipy.fftpack import fft,fftfreq
from scipy.integrate import odeint, RK45


        
class HodgkinHuxley():
    def __init__(self,func,initial_cond,t):
        """Full Hodgkin-Huxley Model implemented in Python"""

        self.C_m  =   1.0
        """membrane capacitance, in uF/cm^2"""

        self.g_Na = 120.0
        """Sodium (Na) maximum conductances, in mS/cm^2"""

        self.g_K  =  36.0
        """Postassium (K) maximum conductances, in mS/cm^2"""

        self.g_L  =   0.3
        """Leak maximum conductances, in mS/cm^2"""

        self.E_Na =  50.0
        """Sodium (Na) Nernst reversal potentials, in mV"""

        self.E_K  = -77.0
        """Postassium (K) Nernst reversal potentials, in mV"""

        self.E_L  = -54.387
        """Leak Nernst reversal potentials, in mV"""

        self.t = t
        """ The time to integrate over """
        
        self.func = func
        self.prev_cond = initial_cond
        self.storeX = [initial_cond]
        """stimulus curve"""
        #self.amp = 10
        """ the magnitude of the square wave"""
        

    def alpha_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.1*(V+40.0)/(1.0 - np.exp(-(V+40.0) / 10.0))

    def beta_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 4.0*np.exp(-(V+65.0) / 18.0)

    def alpha_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.07*np.exp(-(V+65.0) / 20.0)

    def beta_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 1.0/(1.0 + np.exp(-(V+35.0) / 10.0))

    def alpha_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.01*(V+55.0)/(1.0 - np.exp(-(V+55.0) / 10.0))

    def beta_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.125*np.exp(-(V+65) / 80.0)

    def I_Na(self, V, m, h):
        """
        Membrane current (in uA/cm^2)
        Sodium (Na = element name)

        |  :param V:
        |  :param m:
        |  :param h:
        |  :return:
        """
        return self.g_Na * m**3 * h * (V - self.E_Na)

    def I_K(self, V, n):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_K  * n**4 * (V - self.E_K)
    #  Leak
    def I_L(self, V):
        """
        Membrane current (in uA/cm^2)
        Leak

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_L * (V - self.E_L)
    
    def I_inj(self,t):
    
        i = np.where(self.t >= t)[0][0]
        """
        External Current

        |  :param t: time
        |  :return: voltage values relating to stimulus over time.
        
        func = amp*(t>25) - amp*(t>50)
        """
        func = self.func[int(i)-1]
        return func

    @staticmethod
    def dALLdt(X,t,self):
        """
        Integrate

        |  :param X:
        |  :param t:
        |  :return: calculate membrane potential & activation variables
        """
        V, m, h, n = X
        
        dVdt = (self.I_inj(t) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        dmdt = self.alpha_m(V)*(1.0-m) - self.beta_m(V)*m
        dhdt = self.alpha_h(V)*(1.0-h) - self.beta_h(V)*h
        dndt = self.alpha_n(V)*(1.0-n) - self.beta_n(V)*n
        return dVdt, dmdt, dhdt, dndt
    
    def solve(self,t):
        X = odeint(self.dALLdt, self.prev_cond, t, tcrit = t, args = (self,))
        
        V = X[:,0]
        #m = X[:,1]
        #h = X[:,2]
        #n = X[:,3]
        #ina = self.I_Na(V, m, h)
        #ik = self.I_K(V, n)
        #il = self.I_L(V)
        self.prev_cond = [X[-1,0],X[-1,1],X[-1,2],X[-1,3]]
        self.storeX.append([X[-1,0],X[-1,1],X[-1,2],X[-1,3]])
        
        return X
        
    def plot_store(self,n_name):
        """
        Main demo for the Hodgkin Huxley neuron model
        """
        X = np.array(self.storeX)
        V = X[:,0]
        m = X[:,1]
        h = X[:,2]
        n = X[:,3]
        ina = self.I_Na(V, m, h)
        ik = self.I_K(V, n)
        il = self.I_L(V)
        
        fig = plt.figure("Action Potential Graph of {}".format(n_name),figsize=(20,10))
        plt.subplot(4,1,1)
        plt.title('Hodgkin-Huxley Neuron')
        plt.plot(self.t, V, 'k')
        plt.ylabel('V (mV)')
        plt.subplot(4,1,2)
        plt.plot(self.t, ina, 'c', label='$I_{Na}$')
        plt.plot(self.t, ik, 'y', label='$I_{K}$')
        plt.plot(self.t, il, 'm', label='$I_{L}$')
        plt.ylabel('Current')
        plt.legend()

        plt.subplot(4,1,3)
        plt.plot(self.t, m, 'r', label='m')
        plt.plot(self.t, h, 'g', label='h')
        plt.plot(self.t, n, 'b', label='n')
        plt.ylabel('Gating Value')
        plt.legend()

        plt.subplot(4,1,4)
        i_inj_values = [self.I_inj(t) for t in self.t]
        plt.plot(self.t, i_inj_values, 'k')
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
        plt.ylim(-1, 40)
        plt.tight_layout()
        fig.savefig(r'C:\Users\jsalm\Documents\UF\PhD\Spring 2021\BME6938-Neuromechanics\save_bin\{}.png'.format(n_name),dpi=200,bbox_inches='tight')
        plt.close()
        
        return V
 
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
        self.prev_cond = [Xp[-1,0],Xp[-1,1]]
        self.storeX.append([Xp[-1,0],Xp[-1,1]])
        return Xp
        
        
class Muscle(Pendulum):
    def __init__(self,initial_cond,t):
        self.prev_cond = initial_cond
        self.motorunit = 0
        self.storeX = [initial_cond]
        self.t = t
        
    @staticmethod
    def dALLdt(X,t,self):
        tact1 = 100
        beta1 = 0.5
        muscle_act1 = X[0]
                
        dmuscle_actdt = (-1/tact1)*(beta1+(1-beta1)*self.motorunit)*muscle_act1+(1/tact1*self.motorunit)
        return dmuscle_actdt
        
        
    def solve(self,t):
        X = odeint(self.dALLdt, self.prev_cond, t, tcrit = t, args=(self,))
        self.prev_cond = [X[-1,0]]
        self.storeX.append([X[-1,0]])
        
        return X

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
    Ttot = 2
    # total time in second
    f_s = 1000 # sample frequency (samples/s)
    t_s = np.arange(0,Ttot,1/f_s)
    t_ms = np.arange(0,Ttot*f_s,1)
    
        
    def __init__(self):
        self.storeT = [0,0,0]
        self.storeL = [[0,0],[0,0],[0,0]]
        
    def afferent_stim(self,afferentgain,Xp):
    
        theta = Xp[-1,1]
        ang_vel = Xp[-1,0]
        
        if theta > 0:
            n1_amp = abs(afferentgain*(max(ang_vel,0)+1))
            n2_amp = 0
        elif theta < 0:
            n2_amp = abs(afferentgain*(min(0,ang_vel)-1))
            n1_amp = 0
        else:
            n1_amp = 0
            n2_amp = 0
                
        return n1_amp,n2_amp
        
    def efferent_stim(self,neuron1_V):        
        if neuron1_V > -50:
            n1_amp = 1
        else:
            n1_amp = 0
        
        return n1_amp
    
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
        
        #print('length: '+str(Mlength1))
        #print('force: '+str(force1))
        self.storeL.append([Mlength1,Mlength2])
        return force1,force2
        
    def flce_M03(lm=1, lmopt=1, fm0=1, fmmin=0.001, wl=1):
        """McLean (2003) force of the contractile element as function of muscle length.
        
        Parameters
        ----------
        lm  : float, optional (default=1)
            muscle (fiber) length
        lmopt  : float, optional (default=1)
            optimal muscle fiber length
        fm0  : float, optional (default=1)
            maximum isometric muscle force
        fmmin  : float, optional (default=0.001)
            minimum muscle force
        wl  : float, optional (default=1)
            shape factor of the contractile element force-length curve

        Returns
        -------
        fl  : float
            force of the muscle contractile element
        """
        
        fl = np.max([fmmin, fm0*(1 - ((lm - lmopt)/(wl*lmopt))**2)])
    
        return fl
    
    def force_velocity(self,velocity,vmax):
        Fiso = 0.01
        a = .0001
        b = -.5
        invsigmoid = -a/(b+np.exp(-velocity))+Fiso
        #print(invsigmoid)
        return invsigmoid
       
    def muscle_torque(self,muscleact1,muscleact2,Xp,R):
        pendang = Xp[1]
        pendv = Xp[0]
        ### Muscle Params ###
        vmax = .4
        musclegain = 100_00
        a = 1
        c = 1
        dist = .1
        Fres = 0.000001
        ###
        FL_2, FL_1 = self.force_length(a,c,dist,pendang,Fres,R)
        FV = self.force_velocity(pendv,vmax)
        torque = (musclegain/1_000_000*abs(FL_2)*FV*muscleact2)-(musclegain/1_000_000*abs(FL_1)*FV*muscleact1)
        self.storeT.append(torque)
        return torque
        
    def plot_torque(self):
        torque = self.storeT
        length = np.array(self.storeL)
        t = self.t_ms
        
        fig = plt.figure('torque vs time',figsize=(20,10))
        plt.plot(t,torque)
        #plt.plot(t,length[:,0])
        #plt.plot(t,length[:,1])
        plt.xlabel('time (ms)')
        plt.ylabel('torque (N*m)')
        plt.show()
        fig.savefig(r'C:\Users\jsalm\Documents\UF\PhD\Spring 2021\BME6938-Neuromechanics\save_bin\force_graph.png',dpi=200,bbox_inches='tight')
        
    def plot_pend_torq(self,Xp):
        torque = self.storeT
        fig,ax = plt.subplots()
        fig.set_size_inches(18.5,10.5)
        ax.plot(self.t_s,torque,color="red")
        ax.set_xlabel('time (s)')
        ax.set_ylabel('Torque (Nm)',color="red")
        ax2=ax.twinx()
        ax2.plot(self.t_s,Xp[:,1])
        ax2.set_ylabel("Pendulum Angle (rads)",color="blue")
        plt.show()
        fig.savefig(r'C:\Users\jsalm\Documents\UF\PhD\Spring 2021\BME6938-Neuromechanics\save_bin\force_angle_graph.jpg',dpi=200,bbox_inches='tight')
        
    def plot_neuron_pot(self,V1,V2,fname):
        fig,ax = plt.subplots()
        fig.set_size_inches(18.5,10.5)
        ax.plot(self.t_ms,V1,color="red")
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('Membrane Potential of afferent (mV)',color="red")
        ax2=ax.twinx()
        ax2.plot(self.t_ms,V2)
        ax2.set_ylabel('Membrane Potential of efferent (mV)',color="blue")
        plt.show()
        fig.savefig(r'C:\Users\jsalm\Documents\UF\PhD\Spring 2021\BME6938-Neuromechanics\save_bin\{}.jpg'.format(fname),dpi=200,bbox_inches='tight')
        
    def muscle_model(self,affn1,affn2,effn1,effn2,pendulum,muscle1,muscle2,affgain):
        delay_ms = 1 #delay in ms
        time_delay = int(self.f_s/1000*delay_ms) #delay in samples based on sample rate
        print('starting diffeq solver...')
        for i in range(0,len(self.t_ms)-1):
            t_ms = [self.t_ms[i],self.t_ms[i+1]]
            t_s = [self.t_s[i],self.t_s[i+1]]
            
            Xp = pendulum.solve(t_s)
            
            n1_amp,n2_amp = self.afferent_stim(affgain,Xp)
            
            affn1.func.append(n1_amp)
            affn2.func.append(n2_amp)
            
            aff1_X = affn1.solve(t_ms)
            aff2_X = affn2.solve(t_ms)
            
            finder = DomainFinder()
            
            if i > time_delay:
                effn1.func.append(self.efferent_stim(np.array(affn1.storeX)[-(1+time_delay),0])*30)
                effn2.func.append(self.efferent_stim(np.array(affn2.storeX)[-(1+time_delay),0])*30)
                eff1_X = effn1.solve(t_ms)
                eff2_X = effn2.solve(t_ms)
                muscle1.motorunit = self.efferent_stim(eff1_X[-1,0])
                muscle2.motorunit = self.efferent_stim(eff2_X[-1,0])
                
                muscle1_act = muscle1.solve(t_ms)[-1,0]
                muscle2_act = muscle2.solve(t_ms)[-1,0]
                pendulum.force = self.muscle_torque(muscle1_act,muscle2_act,Xp[-(1+time_delay),:],pendulum.R)
                #print(effn1.func[i],np.array(affn1.storeX)[-(1+time_delay),0])
                #print(effn2.func[i],np.array(affn2.storeX)[-(1+time_delay),0])
                
            else:
                n_i = [-65, 0.05, 0.6, 0.32]
                effn1.func.append(0)
                effn2.func.append(0)
                effn1.storeX.append(n_i)
                effn2.storeX.append(n_i)
                muscle1.storeX.append([0])
                muscle2.storeX.append([0])
                pendulum.force = 0

        Xp = pendulum.plot_store()
        V1 = aff_n1.plot_store('afferent 1')
        V2 = aff_n2.plot_store('afferent 2')
        V3 = eff_n1.plot_store('efferent 1')
        V4 = eff_n2.plot_store('efferent 2')
        #self.plot_torque()
        self.plot_pend_torq(Xp)
        self.plot_neuron_pot(V1,V3,'neurons1')
        self.plot_neuron_pot(V2,V4,'neurons2')
    

        
    
if __name__ == "__main__":
    #website: https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/MuscleModeling.ipynb
    MM = Muscle_Mech()
    
    aff_stim1 = []
    aff_stim2 = []
    eff_stim1 = []
    eff_stim2 = []
    n1_i,n2_i,n3_i,n4_i = [-65, 0.05, 0.6, 0.32], [-65, 0.05, 0.6, 0.32],[-65, 0.05, 0.6, 0.32], [-65, 0.05, 0.6, 0.32]
    theta_i = [0,np.pi/8]
    
    aff_n1 = HodgkinHuxley(aff_stim1,n1_i,MM.t_ms)
    aff_n2 = HodgkinHuxley(aff_stim2,n2_i,MM.t_ms)
    
    eff_n1 = HodgkinHuxley(eff_stim1,n1_i,MM.t_ms)
    eff_n2 = HodgkinHuxley(eff_stim2,n2_i,MM.t_ms)        
    
    pend = Pendulum(MM.t_s,theta_i)
    
    muscle1 = Muscle(0,MM.t_ms)
    muscle2 = Muscle(0,MM.t_ms)
    
    MM.muscle_model(aff_n1,aff_n2,eff_n1,eff_n2,pend,muscle1,muscle2,20)
    
    