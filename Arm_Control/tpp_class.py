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

class DbPendulum():
    def __init__(self,n,initial_cond,coeff,mgain,t,f_s):
        self.storeP = [np.hstack(initial_cond)]
        self.storeTx = []
        self.storeTy = []
        self.coeff = coeff #lengths, masses, dampening
        self.n = n
        self.inital_cond = [initial_cond]
        self.prev_cond = np.concatenate([np.broadcast_to(initial_cond[0], n),
                                        np.broadcast_to(initial_cond[1], n)])
        self.gradient = None
        self.t = t
        self.f_s = f_s
        self.g = 0
        self.work = 0
        self.counter = 0
        self.angd = [np.pi/8,-np.pi/8]
        self.musclegain = mgain
        self.storeT = [np.broadcast_to(mgain,n*2)] #store force values
    
    def controller(self,angi,veli,i):
        angd = self.angd[i]
        angj = angi + np.pi/2
        dtheta = angd-angi
        bound = 0.001
        forcex = 0
        forcey = 0
        # if angi < np.pi/2 and i == 0:
            # forcex = self.musclegain*np.cos(angj)*dtheta
        if dtheta > bound and i == 0:
            forcex = self.musclegain*dtheta*np.cos(angj)
            forcey = self.musclegain*dtheta*np.sin(angj)
        elif dtheta < -bound and i == 0:
            forcex = self.musclegain*dtheta*np.cos(angj)
            forcey = self.musclegain*dtheta*np.sin(angj)
        else:
            pass
            
        # if angi > 0 and i == 1:
            # forcex = self.musclegain*np.cos(angj)*dtheta
        if dtheta > bound and i == 1:
            forcex = self.musclegain*dtheta*np.cos(angj)
            forcey = self.musclegain*dtheta*np.sin(angj)
        elif dtheta < -bound and i == 1:
            forcex = self.musclegain*dtheta*np.cos(angj)
            forcey = self.musclegain*dtheta*np.sin(angj)
        else:
            pass
        return (forcex,forcey)
    
    @staticmethod
    def dALLdt(y, t, self, f1, f2):
        """Return the first derivatives of y = theta1, z1, theta2, z2."""
        theta1, z1, theta2, z2 = y
        m1,m2 = self.coeff[1]
        L1,L2 = self.coeff[0]
        g = self.g
        c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)
    
        theta1dot = z1
        z1dot = (f1/m2 + m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
                 (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
        theta2dot = z2
        z2dot = (f1/(m1+m2) + (m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
                 m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
        return theta1dot, z1dot, theta2dot, z2dot

    def solve(self,t,f1,f2):
        Xp = odeint(self.dALLdt, self.prev_cond, t, tcrit = t, args=(self,f1,f2))
        self.prev_cond = Xp[-1,:]
        self.storeP.append(Xp[-1,:])
        self.storeT.append([f1,f2])
        return Xp

#%%
class TpPendulum():
    def __init__(self,n,initial_cond,coeff,mgain,t,f_s):
        self.storeP = [np.hstack(initial_cond)]
        self.storeTx = []
        self.storeTy = []
        self.coeff = coeff #lengths, masses, dampening
        self.n = n
        self.inital_cond = [initial_cond]
        self.prev_cond = np.concatenate([np.broadcast_to(initial_cond[0], n),
                                        np.broadcast_to(initial_cond[1], n)])
        self.gradient = None
        self.t = t
        self.g = 0
        self.f_s = f_s
        self.work = 0
        self.counter = 0
        self.angd = [np.pi/8,-np.pi/8]
        self.musclegain = mgain
        self.storeT = [np.broadcast_to(mgain,n)] #store force values
        
    def controller(self,angi,i):
        # angj = angi + np.pi/2
        angd = self.angd[i]
        dtheta = angd-angi
        bound = 0.001
        force = 0
        # if angi < np.pi/2 and i == 0:
            # forcex = self.musclegain*np.cos(angj)*dtheta
        if dtheta > bound:
            force = -self.musclegain*dtheta
        elif dtheta < -bound:
            force = -self.musclegain*dtheta
        else:
            pass
        return force
        
        
    def create_pendulum(self,times):
        """Integrate a multi-pendulum with `n` sections"""
        lengths = self.coeff[0]
        masses = self.coeff[1]
        dampening = self.coeff[2]
        n = self.n
        g_in = 0
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
        fx1 = mechanics.dynamicsymbols('fx1')
        fx2 = mechanics.dynamicsymbols('fx2')
        fy1 = mechanics.dynamicsymbols('fy1')
        fy2 = mechanics.dynamicsymbols('fy2')
        # fy = symbols('fy:{0}'.format(n))
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
            Ai.set_ang_vel(A, k[i]*u[i]*A.z)
    
            # Create a point in this reference frame
            Pi = P.locatenew('P' + str(i), l[i] * Ai.x)
            Pi.v2pt_theory(P, A, Ai)
    
            # Create a new particle of mass m[i] at this point
            Pai = mechanics.Particle('Pa' + str(i), Pi, m[i])
            particles.append(Pai)
    
            # Set forces & compute kinematic ODE
            if i == 0:
                forces.append((Pi, fx1*A.x+fy1*A.y-m[i]*g*A.y))
            elif i == 1:
                forces.append((Pi, fx2*A.x+fy2*A.y-m[i]*g*A.y))
            kinetic_odes.append(q[i].diff(t) - u[i])
    
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
        damp = np.broadcast_to(dampening,n)
        # Fixed parameters: gravitational , lengths, and masses
        parameters = [g] + list(l) + list(m) + list(k)
        parameter_vals = [g_in] + list(lengths) + list(masses)  + list(damp)
    
        # define symbols for unknown parameters
        dynamic = q + u
        dynamic.append(fx1)
        dynamic.append(fx2)
        dynamic.append(fy1)
        dynamic.append(fy2)
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
            force = np.hstack([self.controller(y[i],y[i+2],i) for i in range(n)])
            fx1 = np.array([force[0]])
            fx2 = np.array([force[2]])
            fy1 = np.array([force[1]])
            fy2 = np.array([force[3]])
            vals = np.concatenate((y, fx1, fx2, fy1, fy2, args))
            sol = np.linalg.solve(mm_func(*vals), fo_func(*vals))
            return np.array(sol).T[0]
        
        def _work_calc(work,counter):
            n = len(self.storeP[counter]) // 2
            if counter > 1:
                dtheta = abs(self.storeP[counter][:n] - self.storeP[counter-1][:n])
                dTor = abs(np.array(self.storeT[counter]) - np.array(self.storeT[counter-1]))
                try:
                    work = work + dTor/dtheta
                except ZeroDivisionError:
                    work = 0
            else:
                work = work
            return work
        
        Xp = odeint(gradient, self.prev_cond, times,tcrit = times, args=(parameter_vals,))
        self.counter += 1
        self.prev_cond = Xp[-1,:]
        self.storeP.append(Xp[-1,:])
        self.storeT.append(np.hstack([self.controller(Xp[-1,i],Xp[-1,i+2],i) for i in range(n)]))
        # self.work = _work_calc(self.work,self.counter)        
        return Xp
    
    @staticmethod
    def dALLdt(y, t, self, f1, f2):
        """Return the first derivatives of y = theta1, z1, theta2, z2."""
        theta1, z1, theta2, z2 = y
        m1,m2 = self.coeff[1]
        L1,L2 = self.coeff[0]
        g = self.g
        c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)
    
        theta1dot = z1
        z1dot = (f1/m2 + m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
                 (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
        theta2dot = z2
        z2dot = (f2/(m1+m2) + (m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
                 m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
        return theta1dot, theta2dot,z1dot, z2dot

    def solve(self,t):
        force = np.hstack([self.controller(self.prev_cond[i],i) for i in range(2)])
        f1 = force[0]
        f2 = force[1]
        Xp = odeint(self.dALLdt, self.prev_cond, t, tcrit = t, args=(self,f1,f2))
        self.prev_cond = Xp[-1,:]
        self.storeP.append(Xp[-1,:])
        self.storeT.append([f1,f2])
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
class Muscle_Mech():
    def __init__(self,musclegain,f_s,t):
        # n = 3
        self.storeT = []
        self.storeL = []
        self.storeP = []
        self.dtheta = []
        self.musclegain = musclegain
        self.f_s = f_s
        self.t = t
    
    def arm_model(self,pendulum,ang_desired,plot=True):
        n = pendulum.n
        muscle_forces = np.hstack([(0,0) for i in range(n)])
        print('starting diffeq solver...')
        self.storeT.append(muscle_forces)
        for i in range(0,len(self.t)-1):
            t_s = [self.t[i],self.t[i+1]]            
            pendulum.solve(t_s)
        if plot:
            # pendulum.plot_pendulum_trace()
            self.plot_pend_torq(pendulum.storeT,np.stack(pendulum.storeP)[:,:n],ang_desired)
            plt.waitforbuttonpress()
        else:
            return 0
        
    def plot_torque_spring(self,spring):
        torque = np.stack(self.storeT)
        length = np.stack(spring.storeX)[:,0]
        fig,ax = plt.subplots()
        plt.title('plot_torque_length')
        fig.set_size_inches(18.5,10.5)
        ax.plot(self.t,torque,color="red",alpha = 0.5,label='torque')
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('Force (N)',color="red")
        ax2=ax.twinx()
        ax2.plot(self.t,length,alpha = 0.7, label='Length')
        ax2.set_ylabel("Hopping Height (m)",color="blue")
        plt.show()
        fig.savefig(os.path.join(save_bin,'force_length_graph.png'),dpi=200,bbox_inches='tight')
        
    def plot_torque(self):
        torque = np.stack(self.storeT)
        length = np.stack(self.storeL)[:,0]
        fig,ax = plt.subplots()
        plt.title('plot_torque_length')
        fig.set_size_inches(18.5,10.5)
        ax.plot(self.t,torque,color="red",label='torque')
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('Torque (Nm)',color="red")
        ax2=ax.twinx()
        ax2.plot(self.t,length,label='Length')
        ax2.set_ylabel("Muscle Length (m)",color="blue")
        plt.show()
        fig.savefig(os.path.join(save_bin,'force_length_graph.png'),dpi=200,bbox_inches='tight')
    
    def plot_pend_torq(self,storeT,Xp,ang_desired):
        torque = np.stack(storeT)
        fig,ax = plt.subplots()
        plt.title('plot_torque_pend')
        fig.set_size_inches(18.5,10.5)
        ax.plot(self.t,torque,color="red",label='torque')

        ax.set_xlabel('time (ms)')
        ax.set_ylabel('Torque (Nm)',color="red")
        ax2=ax.twinx()
        ax2.plot(self.t,Xp,color="blue",label='pendulum angle')
        ax2.plot(self.t,np.ones(Xp.shape[0])*ang_desired[0])
        ax2.plot(self.t,np.ones(Xp.shape[0])*ang_desired[1])
        ax2.set_ylabel("Pendulum Angle (rads)",color="blue")
        plt.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig(os.path.join(save_bin,'force_angle_graph.jpg'),dpi=200,bbox_inches='tight')
    

#%%
if __name__ == "__main__":
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
    #mech
    mgain = 5
    ### Time ###
    Ttot = 5 # total time in second
    f_s = 200 # sample frequency (samples/s)
    t_s = np.arange(0,Ttot,1/f_s)
    t_ms = np.arange(0,Ttot*f_s,1)
    
    initial_cond_tpp = [pos1,vel,pos2,vel]
    coeff_tpp = [[l1,l2],[m1,m2],damp]
    
    ### INIT ###
    Arm = TpPendulum(n,initial_cond_tpp,coeff_tpp,mgain,t_s,f_s)
    mech = Muscle_Mech(mgain,f_s,t_s)
    
    for i in range(0,len(t_s)-1):
        t = [t_s[i],t_s[i+1]]
        f1 = 0
        f2 = 0
        Xp = Arm.solve(t)
    'end for'
    X = np.stack(Arm.storeP)
    T = np.stack(Arm.storeT)
    Arm.plot_pendulum_trace()
    anim = Arm.animate_pendulum()
    
    