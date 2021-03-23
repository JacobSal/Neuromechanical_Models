import sys, os
sys.path.append(r'C:\Users\jsalm\Documents\UF\PhD\Spring 2021\BME6938-Neuromechanics\Berkely Modanna\Py Mimicks')

import numpy as np
import scipy as sp
import pylab as plt
import os
from scipy.fftpack import fft,fftfreq
from scipy.integrate import odeint, RK45
from scipy.signal import find_peaks

from sympy import symbols
from sympy.physics import mechanics

from sympy import Dummy, lambdify

#animation functions
from matplotlib import animation

dirname = os.path.dirname(__file__)
save_bin = os.path.join(dirname,"save_bin")

#%%       
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

#%%
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

#%%
class TpPendulum():
    def __init__(self,n,initial_cond,coeff,t,f_s):
        self.storeP = []
        self.storeT = []
        self.coeff = coeff #lengths, masses, dampening
        self.n = n
        self.inital_cond = [initial_cond]
        self.prev_cond = np.concatenate([np.broadcast_to(initial_cond[0], n),
                                        np.broadcast_to(initial_cond[1], n)])
        self.gradient = None
        self.storeP = []
        self.t = t
        self.f_s = f_s
        self.work = 0
        self.counter = 0
        
    def create_pendulum(self,times,exforce):
        """Integrate a multi-pendulum with `n` sections"""
        lengths = self.coeff[0]
        masses = self.coeff[1]
        dampening = self.coeff[2]
        n = self.n
        self.storeT.append(exforce)
        #-------------------------------------------------
        # Step 1: construct the pendulum model
        
        # Generalized coordinates and velocities
        # (in this case, angular positions & velocities of each mass) 
        q = mechanics.dynamicsymbols('q:{0}'.format(n))
        u = mechanics.dynamicsymbols('u:{0}'.format(n))
    
        # mass and length and dampening
        m = symbols('m:{0}'.format(n))
        l = symbols('l:{0}'.format(n))
        k = symbols('k:{0}'.format(n))
    
        # gravity and time symbols
        g, t = symbols('g,t')
        
        #force
        f = symbols('f:{0}'.format(n))
        
        #--------------------------------------------------
        # Step 2: build the model using Kane's Method
    
        # Create pivot point reference frame
        A = mechanics.ReferenceFrame('A')
        P = mechanics.Point('P')
        P.set_vel(A, 0)
    
        # lists to hold particles, forces, and kinetic ODEs
        # for each pendulum in the chain
        particles = []
        forces = []
        kinetic_odes = []
    
        for i in range(n):
            # Create a reference frame following the i^th mass
            Ai = A.orientnew('A' + str(i), 'Axis', [q[i], A.z])
            Ai.set_ang_vel(A, u[i] * A.z)
    
            # Create a point in this reference frame
            Pi = P.locatenew('P' + str(i), l[i] * Ai.x)
            Pi.v2pt_theory(P, A, Ai)
    
            # Create a new particle of mass m[i] at this point
            Pai = mechanics.Particle('Pa' + str(i), Pi, m[i])
            particles.append(Pai)
    
            # Set forces & compute kinematic ODE
            forces.append((Pi, f[i]*A.x-m[i]*g*A.y))
            kinetic_odes.append(q[i].diff(t) - u[i]*k[i])
    
            P = Pi
    
        # Generate equations of motion
        KM = mechanics.KanesMethod(A, q_ind=q, u_ind=u,
                                   kd_eqs=kinetic_odes)
        fr, fr_star = KM.kanes_equations(particles, forces)
        
        #-----------------------------------------------------
        # Step 3: numerically evaluate equations and integrate
    
        # initial positions and velocities – assumed to be given in degrees
            
        # lengths and masses
        if lengths is None:
            lengths = np.ones(n) / n
        lengths = np.broadcast_to(lengths, n)
        masses = np.broadcast_to(masses, n)
        exforces = np.broadcast_to(exforce,n)
        damp = np.broadcast_to(dampening,n)
    
        # Fixed parameters: gravitational , lengths, and masses
        parameters = [g] + list(l) + list(m) + list(f) + list(k)
        parameter_vals = [9.81] + list(lengths) + list(masses) + list(exforces) + list(damp)
    
        # define symbols for unknown parameters
        dynamic = q + u
        unknowns = [Dummy() for i in dynamic]
        unknown_dict = dict(zip(dynamic, unknowns))
        kds = KM.kindiffdict()
        
        # substitute unknown symbols for qdot terms
        mm_sym = KM.mass_matrix_full.subs(kds).subs(unknown_dict)
        fo_sym = KM.forcing_full.subs(kds).subs(unknown_dict)
    
        # create functions for numerical calculation
        mm_func = lambdify(unknowns + parameters, mm_sym)
        fo_func = lambdify(unknowns + parameters, fo_sym)
        
        def gradient(y, t, args):
            vals = np.concatenate((y, args))
            sol = np.linalg.solve(mm_func(*vals), fo_func(*vals))
            return np.array(sol).T[0]
        
        def _work_calc(work,counter):
            n = len(self.storeP[counter]) // 2
            if counter > 1:
                dtheta = abs(self.storeP[counter][:n] - self.storeP[counter-1][:n])
                dTor = abs(np.array(self.storeT[counter]) - np.array(self.storeT[counter-1]))
                work = work + dTor/dtheta
            else:
                work = work
            return work
        
        Xp = odeint(gradient, self.prev_cond, times,tcrit = times, args=(parameter_vals,))
        self.work = _work_calc(self.work,self.counter)        
        self.counter += 1
        self.prev_cond = Xp[-1,:]
        self.storeP.append(Xp[-1,:])
        return Xp
    
    def _conv_ang(Xp):
        return Xp + np.pi/2
        
        
    def get_xy_coords(self):
        """Get (x, y) coordinates from generalized coordinates p"""
        p = np.atleast_2d(np.stack(self.storeP))
        n = p.shape[1] // 2
        lengths = np.array(self.coeff[0])
        if lengths is None:
            lengths = np.ones(n) / n
        zeros = np.zeros(p.shape[0])[:, None]
        x = np.hstack([zeros, lengths * np.cos(p[:, :n])])
        y = np.hstack([zeros, lengths * np.sin(p[:, :n])])
        return np.cumsum(x, 1), np.cumsum(y, 1)
    
    def plot_pendulum_trace(self):
        plt.close("triple Pendulum Trace")
        x, y = self.get_xy_coords()
        lim = max(self.coeff[0])*self.n
        plt.figure("triple Pendulum Trace")
        plt.plot(x, y);
        plt.xlim(-lim,lim)
        plt.ylim(-lim,lim)
        plt.xlabel("position (m)")
        plt.ylabel("position (m)")
        plt.show()
        # plt.close()
        return 0
    
    def set_new_tpp(self,Xp,n_p):
        n = Xp.shape[1] // 2
        n_p_new = n_p.copy()
        n_p_new[0] = list(Xp[-1,:n])
        n_p_new[1] = list(Xp[-1,n:])
        return n_p_new
        
    def animate_pendulum(self,):
        x, y = self.get_xy_coords()
        lim = max(self.coeff[0])*self.n
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim))
    
        line, = ax.plot([], [], 'o-', lw=2)
    
        def init():
            line.set_data([], [])
            return line,
    
        def animate(i):
            line.set_data(x[i], y[i])
            return line,
    
        anim = animation.FuncAnimation(fig, animate, frames=len(self.t),
                                       interval=self.f_s * self.t.max() / len(self.t),
                                       blit=True, init_func=init)
        # plt.close(fig)
        return anim

#%%
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
            foft = func*self.springgain
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
    

#%%
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
 
#%%       
class Muscle_Mech(DomainFinder):
    Ttot = 5
    # total time in second
    f_s = 200 # sample frequency (samples/s)
    t_s = np.arange(0,Ttot,1/f_s)
    t_ms = np.arange(0,Ttot*f_s,1)
    
    def __init__(self,musclegain):
        # n = 3
        self.storeT = []
        self.storeL = []
        self.storeP = []
        self.musclegain = musclegain
        
    def stim(self,x1,x2):        
        y1=max(x1,0)
        y2=max(x2,0)
        y=y1-y2
        return (y1,y2,y)
    
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
        # self.storeT.append(torque)
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
            
    def hoping_model(self,matsuoka,springmass,affgain,n_i,coeff,plot=False):
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
                # self.storeT.append(0)
            # judging steady states using average frequency of top 5 peaks
            # if ub > 500:
            #     bb += 1 
            #     avgfrq.append([self._avg_freq_profile(np.stack(matsuoka.storeX)[bb:ub,0],5),
            #                    self._avg_freq_profile(np.stack(matsuoka.storeX)[bb:ub,1],5)])
        if plot:
            Xp = springmass.plot_store()
            matsuoka.plot_store(self.t_ms,'coeff_{}'.format(coeff),"Mass Height (m)")
            self.plot_torque_spring(springmass)
            plt.waitforbuttonpress()
        else:
            return 0
    
    def three_pend_model(self,matsuoka,pendulum,n_i,n_p,n=3,plot=True):
        """
        

        Parameters
        ----------
        matsuoka1 : obj
            DESCRIPTION.
        matsuoka2 : obj
            DESCRIPTION.
        matsuoka3 : obj
            DESCRIPTION.
        n_i : list of ints
            DESCRIPTION.
        n_p : list of lists
            n_p[0] = initial position, n_p[1] = initial_velocities for n segments. 
            n_p[2] = lengths, n_p[3] = masses, n_p[4] = dampening

        Returns
        -------
        None.

        """
        delay_ms = 1 #delay in ms
        time_delay = int(self.f_s/1000*delay_ms) #delay in samples based on sample rate
        print('starting diffeq solver...')
        ub = 0
        bb = 0
        count = 0
        muscle_forces = np.broadcast_to(0,n)
        avgfrq = []
        self.storeT.append([0,0,0])
        pendulum.storeP.append(np.array(n_p).reshape(n*2,))
        for i in range(0,len(self.t_ms)-1):
            #Initia time and Pendulum diffeq
            t_ms = [self.t_ms[i],self.t_ms[i+1]]
            t_s = [self.t_s[i],self.t_s[i+1]]            
            Xp = pendulum.create_pendulum(t_s,muscle_forces)
            ang = TpPendulum._conv_ang(Xp[-1,:n])
            ub += 1
            #delay iteration of force from muscle and activation of neurons
            if i > time_delay:
                mats_X1 = [matsuoka[i].solve(t_ms,ang[i]) for i in range(n)] 
                y = [self.stim(mats_X1[i][-1,0],mats_X1[0][-1,1]) for i in range(len(mats_X1))]
                muscle_forces = [self.muscle_torque(y[i][2]) for i in range(len(mats_X1))]
                self.storeT.append(muscle_forces)
            else:
                #default storage
                [matsuoka[i].storeX.append(n_i) for i in range(n)]
                [matsuoka[i].storeoc.append(ang[i]) for i in range(n)]
                self.storeT.append(muscle_forces)    
            #judging steady states using average frequency of top 5 peaks
            # if ub > 500:
            #     bb += 1 
            #     avgfrq1.append(self._avg_freq_profile(np.stack(matsuoka.storeX)[bb:ub,0],5,self.f_s))
            #     avgfrq2.append(self._avg_freq_profile(np.stack(matsuoka.storeX)[bb:ub,1],5,self.f_s))
            #     if abs(avgfrq1[-2]-avgfrq2[-1]) < 0.1 and count == 0:
            #         print("reached steady state")    
        if plot:
            pendulum.plot_pendulum_trace()
            [matsuoka[i].plot_store(self.t_ms,'coeff_{0}_N{1}'.format(n_i,i),"Mass Angle (rad)") for i in range(n)]
            self.plot_pend_torq(TpPendulum._conv_ang(np.stack(pendulum.storeP)[:,:n]))
            plt.waitforbuttonpress()
        else:
            return 0
    
    def plot_torque_spring(self,spring):
        torque = np.stack(self.storeT)
        length = np.stack(spring.storeX)[:,0]
        fig,ax = plt.subplots()
        plt.title('plot_torque_length')
        fig.set_size_inches(18.5,10.5)
        ax.plot(self.t_ms,torque,color="red",alpha = 0.5,label='torque')
        ax.set_xlabel('time (ms)')
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
        ax.set_xlabel('time (ms)')
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
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('Torque (Nm)',color="red")
        ax2=ax.twinx()
        ax2.plot(self.t_ms,Xp,label='pendulum angle')
        ax2.set_ylabel("Pendulum Angle (rads)",color="blue")
        plt.show()
        fig.savefig(os.path.join(save_bin,'force_angle_graph.jpg'),dpi=200,bbox_inches='tight')
    

#%%
if __name__ == "__main__":
    #website: https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/MuscleModeling.ipynb
    coeff = [2,2.5,12,12,55,55*3,6] #bc, wc, h1, h2, t1, t2, c1
    pos = -np.pi/4
    vel = 1
    pendfreq = 0.1 #set this
    radfreq = pendfreq*2*np.pi
    l1 = 9.81/1000/(radfreq**2)
    m1 = .1
    initial_cond_mats = np.array([np.pi/7,np.pi/14,0,0])
    initial_cond_tpp = [[pos,pos*.9,pos*.8],[vel,vel*.8,vel*.9]]
    coeff_tpp = [[l1,l1,l1],[m1,m1,m1],1]
    mgain = 0.01
    n = 3
    
    mech = Muscle_Mech(mgain)
    mats = [Matsuoka(initial_cond_mats,coeff,TpPendulum._conv_ang(initial_cond_tpp[0][i])) for i in range(3)]
    pend = TpPendulum(n,initial_cond_tpp,coeff_tpp,mech.t_s,mech.f_s)
    mech.three_pend_model(mats,pend,initial_cond_mats,initial_cond_tpp)
    plt.close('all')
    anim = pend.animate_pendulum()
    

    #X = mats.plot_store(t_ms,'matsuoka model')
    
    