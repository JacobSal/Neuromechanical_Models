import sys, os
sys.path.append(r'C:\Users\jsalm\Documents\UF\PhD\Spring 2021\BME6938-Neuromechanics\Berkely Modanna\Py Mimicks')
"new commit"
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
class Ball():
    def __init__(self,time,e,mass,initial_cond):
        self.t = time
        self.e = e #coefficient of restitution
        self.m = mass
        self.prev_cond = initial_cond
        self.storeX = [initial_cond]
        self.g = 9.81 #m/s**2
        self.storeB = []
        self.storeF = []
    
    @staticmethod
    def dALLdt(X,t,self):
        h,v = X
        if int(h) > 0:
            dhdt = v
            dvdt = -self.g
        elif int(h) <= 0:
            dhdt = 0
            dvdt = -self.e*dhdt
        return dhdt,dvdt
    
    def solve(self,t):
        Xp = odeint(self.dALLdt, self.prev_cond, t, tcrit = t, args=(self,))
        self.prev_cond = Xp[-1,:]
        self.storeB.append(Xp[-1,:])
        return Xp 
    def calc_theta(self,x0,y0,x1,y1):
        return np.arcsin((x1-x0)/np.sqrt((x1-x0)**2+(y1-y0)**2))
    
    def calc_force(self,vel,dt):
        return vel/dt*self.m
    
    def simple_ball(self,vel0,x0,y0,dt,theta,obstacle):
        """
        vel0: Type int
            initial velocity in the horizontal plane
        y0: Type int
            initial position in the verticla axis
        x0: Type int
            initial position in the horizontal axis
        e: Type float
            elasticity parameter
        theta: Type float (Rads)
            angle at which the ball hits the obstacle
        obstacle: Type list of tuples
        """
        airres = 1
        vel = vel0*airres
        x = vel*dt+x0
        y = -9.81*dt*dt+y0
        close = np.array([(x0-x,y0-y) for x,y in obstacle])
        if np.any(abs(close)<0.01) or y0<=0: #pretty close so lets just stick the ball to the obstacle (perfect catch)
            vel = 0
            x,y = x0,y0
        #     theta = self.calc_theta(x0,y0,xf,yf)
        #     vel = -self.e*vel0
        #     x = vel*dt*np.cos(theta) + x0
        #     y = -9.81*dt*dt+vel*dt*np.sin(theta) + y0
        # else:
        #     vel = vel0
        #     x = vel*dt+x0
        #     y = -9.81*dt*dt+y0
        self.storeB.append([vel,x,y])
        return vel,x,y
    
    def plot_ball_trace(self):
        plt.close("ball Trace")
        x = np.stack(self.storeB)[:,1]
        y = np.stack(self.storeB)[:,2]
        lim = np.max(np.stack(self.storeB)[:,1:])
        plt.figure("ball Trace")
        plt.plot(x, y);
        plt.xlim(-lim-.5,lim+.5)
        plt.ylim(-lim-.5,lim+.5)
        plt.xlabel("position (m)")
        plt.ylabel("position (m)")
        plt.show()
        # plt.close()
        plt.savefig(os.path.join(save_bin,'xy_trace_ball.png'),dpi=200,bbox_inches='tight')
        return 0
            
            
        
        

#%%
class TpPendulum():
    def __init__(self,n,initial_cond,coeff,mgain,t,f_s):
        self.storeP = [np.hstack(initial_cond)]
        self.coeff = coeff #lengths, masses, dampenin
        self.inital_cond = [initial_cond]
        self.prev_cond = np.concatenate([np.broadcast_to(initial_cond[0], n),
                                        np.broadcast_to(initial_cond[1], n)])
        self.gradient = None
        self.t = t
        self.g = 0
        self.f_s = f_s
        self.work = 0
        self.angd = [np.pi/8,-np.pi/8]
        self.musclegain = mgain
        self.storeT = [np.broadcast_to(0,2)] #store force values
        x,y = self.get_xy()
        self.x = x[0,2]
        self.y = y[0,2]
    def calc_theta(self,x0,y0,x1,y1):
        return np.arcsin((x1-x0)/np.sqrt((x1-x0)**2+(y1-y0)**2))
    
    def get_xy(self):
        """Get (x, y) coordinates from generalized coordinates p"""
        p = np.atleast_2d(self.prev_cond)
        n = p.shape[1] // 2
        lengths = np.array(self.coeff[0])
        if lengths is None:
            lengths = np.ones(n) / n
        zeros = np.zeros(p.shape[0])[:, None]
        x = np.hstack([zeros, lengths * np.sin(p[:, :n])])
        y = np.hstack([zeros, -lengths * np.cos(p[:, :n])])
        return np.cumsum(x, 1), np.cumsum(y, 1)
    
    def controller1(self,angi,i):
        # angj = angi + np.pi/2
        angd = self.angd[i]
        dtheta = angd-angi
        bound = 0.001
        force = 0
        # if angi < np.pi/2 and i == 0:
            # forcex = self.musclegain*np.cos(angj)*dtheta
        if dtheta > bound:
            force = self.musclegain*dtheta
        elif dtheta < -bound:
            force = self.musclegain*dtheta
        else:
            pass
        return force
    
    def controller2(self,x1,y1,i):
        #look in blue book: determine the ball angle to the origin and reduce that angle of the arm
        #relative to the ball angle from origin.
        # angj = angi + np.pi/2
        x,y = self.get_xy()
        dist = np.sqrt((x1-x[0,i+1])**2+(y1-y[0,i+1])**2)
        bound = 0.001
        force = 0
        if dist > bound:
            force = self.musclegain*-dist
        elif dist < -bound:
            force = self.musclegain*dist
        else:
            pass
        return force
    
    @staticmethod
    def dALLdt(y, t, self, f1, f2):
        """Return the first derivatives of y = theta1, z1, theta2, z2."""
        theta1, theta2, z1 , z2 = y
        m1,m2 = self.coeff[1]
        L1,L2 = self.coeff[0]
        k = self.coeff[2]
        g = self.g
        c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)
        
        theta1dot = z1
        if theta1<np.pi/2:
            z1dot = (f1/m2 - k*theta1dot  + m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
                 (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
        else:
            z1dot = -k*theta1dot
        theta2dot = z2
        if theta2 < 0: 
            z2dot = (f2/m2 - k*theta2dot + (m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
                 m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
        else:
            z2dot = -k*theta2dot
            
        return theta1dot,theta2dot,z1dot, z2dot

    def solve(self,t,x0,y0):
        force = [self.controller2(x0,y0,i) for i in range(0,2)]
        f1 = force[0]
        f2 = force[1]
        Xp = odeint(self.dALLdt, self.prev_cond, t, tcrit = t, args=(self,f1,f2))
        self.prev_cond = Xp[-1,:]
        self.storeP.append(Xp[-1,:])
        self.storeT.append([f1,f2])
        return Xp        
        
    def get_xy_coords(self):
        """Get (x, y) coordinates from generalized coordinates p"""
        p = np.atleast_2d(np.stack(self.storeP))
        n = p.shape[1] // 2
        lengths = np.array(self.coeff[0])
        if lengths is None:
            lengths = np.ones(n) / n
        zeros = np.zeros(p.shape[0])[:, None]
        x = np.hstack([zeros, lengths * np.sin(p[:, :n])])
        y = np.hstack([zeros, -lengths * np.cos(p[:, :n])])
        return np.cumsum(x, 1), np.cumsum(y, 1)
    
    def plot_pendulum_trace(self):
        plt.close("triple Pendulum Trace")
        x, y = self.get_xy_coords()
        lim = max(self.coeff[0])*2
        plt.figure("triple Pendulum Trace")
        plt.plot(x, y);
        plt.xlim(-lim,lim)
        plt.ylim(-lim,lim)
        plt.xlabel("position (m)")
        plt.ylabel("position (m)")
        plt.show()
        # plt.close()
        plt.savefig(os.path.join(save_bin,'xy_trace.png'),dpi=200,bbox_inches='tight')
        return 0
    
    def set_new_tpp(self,Xp,n_p):
        n = Xp.shape[1] // 2
        n_p_new = n_p.copy()
        n_p_new[0] = list(Xp[-1,:n])
        n_p_new[1] = list(Xp[-1,n:])
        return n_p_new
        
    def animate_pendulum(self,):
        x, y = self.get_xy_coords()
        lim = max(self.coeff[0])*2
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
        n = 2
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
        ax2.plot(self.t,Xp[:,0],color="blue",label='pendulum angle 1')
        ax2.plot(self.t,Xp[:,1],color="orange",label='pendulum angle 2')
        ax2.plot(self.t,np.ones(Xp.shape[0])*ang_desired[0],color="blue", label = 'upper bound')
        ax2.plot(self.t,np.ones(Xp.shape[0])*ang_desired[1],color="orange", label = 'lower bound')
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
    mgain = 100
    #ball
    e = 1 #coefficient of restitution
    mass = 2
    obstacle = []
    vel = 3
    y = 1
    x = -1
    theta = 0
    ### Time ###
    Ttot = 5 # total time in second
    f_s = 500 # sample frequency (samples/s)
    t_s = np.arange(0,Ttot,1/f_s)
    t_ms = np.arange(0,Ttot*f_s,1)
    
    initial_cond_tpp = [[pos1,pos2],[vel,vel]]
    coeff_tpp = [[l1,l2],[m1,m2],damp]
    initial_cond_ball = [10,0]
    
    ### INIT ###
    Arm = TpPendulum(n,initial_cond_tpp,coeff_tpp,mgain,t_s,f_s)
    mech = Muscle_Mech(mgain,f_s,t_s)
    Ball = Ball(t_s,e,mass,initial_cond_ball)
    
    for i in range(0,len(t_s)-1):
        t = [t_s[i],t_s[i+1]]
        dt = abs(t_s[i+1] - t_s[i])
        vel,x,y = Ball.simple_ball(vel,x,y,dt,theta,obstacle)
        Xp = Arm.solve(t,x,y)
        obstacle = [(Arm.x,Arm.y)]
        # Bp = ball.solve(t)
    'end for'
    X = np.stack(Arm.storeP)
    T = np.stack(Arm.storeT)
    B = np.stack(Ball.storeB)
    Arm.plot_pendulum_trace()
    Ball.plot_ball_trace()
    anim = Arm.animate_pendulum()
        