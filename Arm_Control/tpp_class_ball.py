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
from scipy.spatial import distance

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
        self.e = .7 #coefficient of restitution
        self.m = mass
        self.prev_cond = initial_cond
        self.storeX = [initial_cond]
        self.g = 9.81 #m/s**2
        self.storeB = [initial_cond]
        self.storeF = []
        self.storeC = [0]
    
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
    
    def simple_ball(self,vx,vy,x0,y0,dt,theta,obstacle):
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
        ballcatch = False
        airres = 1
        x = vx*airres*dt+x0
        y = vy*airres*dt + y0
        vy = -self.g*dt+vy
        close = np.sqrt((x0-obstacle[0])**2+(y0-obstacle[1])**2)
        self.storeC.append(close)
        if np.any(abs(close)<0.08) or ballcatch: #pretty close so lets just stick the ball to the obstacle (perfect catch)
            vx,vy = 0,0
            x,y = x0,y0
            ballcatch = True
        if y < -1:
            y = -1
            vy = -vy*self.e
        self.storeB.append([vx,vy,x,y])
        return vx,vy,x,y,ballcatch
    
    def plot_ball_trace(self):
        plt.close("ball Trace")
        x = np.stack(self.storeB)[:,2]
        y = np.stack(self.storeB)[:,3]
        lim = np.max(np.stack(self.storeB)[:,2:])
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
# def resettable(f):
#     import copy

#     def __init_and_copy__(self, *args, **kwargs):
#         f(self, *args)
#         self.__original_dict__ = copy.deepcopy(self.__dict__)

#         def reset(o = self):
#             o.__dict__ = o.__original_dict__

#         self.reset = reset

#     return __init_and_copy__

class TpPendulum(object):
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
        self.musclegain = mgain
        self.fprev = [mgain*1,mgain*1]
        self.storeT = [np.broadcast_to(0,2)] #store force values
        

    def calc_theta(self,x0,y0,x1,y1):
        return np.arcsin((x1-x0)/np.sqrt((x1-x0)**2+(y1-y0)**2))
    
    def get_theta(self,x,y):
        if x == 0 and y > 0:
            theta = np.pi/2
        elif x == 0 and y < 0:
            theta = -np.pi/2
        elif x > 0:
            theta = np.arctan(y/x)
        elif x < 0:
            theta = np.pi+np.arctan(y/x)
        return theta
    

        
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
    
    def controller1(self,vel,x0,y0,i):
        # feedforward
        l1,l2 = self.coeff[0]
        ArmRadius = l1 + l2        
        xA,yA = self.get_xy()
        
        def get_BallVector(t,x0,y0,vel0):
            airres = 1
            error = 0
            vel = vel0*airres
            x = vel*t+x0+error/x0
            y = -9.81*t**2/2+vel*t+y0+error/y0
            return (x,y)
        
        predictBall = [get_BallVector(t,x0,y0,vel) for t in self.t]
        armRadii = [(ArmRadius*np.cos(theta),ArmRadius*np.sin(theta)) for theta in np.arange(0,2*np.pi,2*np.pi/len(predictBall))]
        store = distance.cdist(predictBall,armRadii,'euclidean')
        val = np.argwhere(store == np.min(store))[0]
        time = self.t[val[0]]
        # xB,yB = predictBall[val[0]]
        # diff = np.sqrt((xB-xA)**2+(yB-yA)**2)
        # force = None
        return time
    
    def controller1_err(self):
        
        pass
    
    def controller2(self,x1,y1,i):
        #Feedback model
        # angj = angi + np.pi/2        
        x,y = self.get_xy()
        thA = self.get_theta(x[0,i+1],y[0,i+1])
        thB = self.get_theta(x1,y1)
        diff = thB-thA
        dist = diff*np.sqrt((x1-x[0,i+1])**2+(y1-y[0,i+1])**2)
        force = self.musclegain*dist
        return force
    
    # @staticmethod
    # def dALLdt(y, t, self, f1, f2, ballcatch):
    #     """Return the first derivatives of y = theta1, z1, theta2, z2."""
    #     theta1, theta2, z1 , z2 = y
    #     m1,m2 = self.coeff[1]
    #     L1,L2 = self.coeff[0]
    #     k = self.coeff[2]
    #     g = self.g
    #     c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)
        
    #     if ballcatch:
    #         theta1dot = 0
    #         theta2dot = 0
    #         z1dot = 0
    #         z2dot = 0
    #     else:
    #         theta1dot = z1
    #         if theta1<np.pi/2:
    #             z1dot = (f1/m2 - k*theta1dot  + m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
    #                  (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    #         else:
    #             z1dot = -k*theta1dot
                
    #         theta2dot = z2
    #         if theta2 < 0: 
    #             z2dot = (f2/m2 - k*theta2dot + (m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
    #                  m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
                
    #         else:
    #             z2dot = -k*theta2dot 
        
    #     return theta1dot,theta2dot,z1dot,z2dot
    
    @staticmethod
    def dALLdt(y, t, self, f1, f2, ballcatch):
        """Return the first derivatives of y = theta1, z1, theta2, z2."""
        theta1, theta2, z1 , z2 = y
        m1,m2 = self.coeff[1]
        L1,L2 = self.coeff[0]
        k = self.coeff[2]
        g = self.g
        c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)
        
        if ballcatch:
            theta1dot = 0
            theta2dot = 0
            z1dot = 0
            z2dot = 0
        else:
            theta1dot = z1
            z1dot = (f1/m2 - k*theta1dot  + m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
                 (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
                
            theta2dot = z2
            z2dot = (f2/m2 - k*theta2dot + (m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
                 m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
        
        return theta1dot,theta2dot,z1dot,z2dot

    def fbsolve(self,t,x0,y0,ballcatch):
        force = [self.controller2(x0,y0,i) for i in range(0,2)]
        f1 = force[0]
        f2 = force[1]
        Xp = odeint(self.dALLdt, self.prev_cond, t, tcrit = t, args=(self,f1,f2,ballcatch))
        self.prev_cond = Xp[-1,:]
        self.storeP.append(Xp[-1,:])
        self.storeT.append([f1,f2])
        return Xp
    
    def ffsolve(self,t,vx,vy,x0,y0,f1,f2,ballcatch):
        Xp = odeint(self.dALLdt, self.prev_cond, t, tcrit = t, args=(self,f1,f2,ballcatch))
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
        # ax.axis('off')
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
    
    def animate_ball_pendulum(self,ballxy):
        x, y = self.get_xy_coords()
        x0,y0 = ballxy[0,:]
        lim = 3 #max(self.coeff[0])
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        # ax.axis('off')
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim))    
        line, = ax.plot([], [], 'o-', lw=2)
        ball = plt.Circle((x0,y0), 0.08)
        ax.add_patch(ball)
    
        def init():
            line.set_data([], [])
            ball.set_center((x0,y0))            
            return line, ball,
    
        def animate(i):
            line.set_data(x[i], y[i])
            ball.set_center((ballxy[i,0],ballxy[i,1]))
            return line, ball,
    
        anim = animation.FuncAnimation(fig, animate, frames=len(self.t),
                                       interval=self.f_s * self.t.max() / len(self.t),
                                       blit=True, init_func=init)
        plt.show()
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
        
    def ff_iterator(self,Arm,Ball,maxiter,vx,vy,x,y,f1adj,f2adj,ballcatch):
        xA,yA = Armobj.get_xy()
        obstacle = (xA[0,2],yA[0,2])
        f1 = Arm.musclegain+f1adj
        f2 = Arm.musclegain+f2adj
        for i in range(0,len(t_s)-1):
            t = [t_s[i],t_s[i+1]]
            dt = abs(t_s[i+1] - t_s[i])
            vx,vy,x,y,ballcatch = Ball.simple_ball(vx,vy,x,y,dt,theta,obstacle) #run ball dynamics
            Xp = Arm.ffsolve(t,vx,vy,x,y,f1,f2,ballcatch) #use pendulum differential equation to judge movement
            xA,yA = Armobj.get_xy()
            obstacle = (xA[0,2],yA[0,2]) #produce obstacle from the peripheral end of pendulum
        'end for'
        print(ballcatch)
        tempB = np.stack(Ball.storeB)[:,2:]
        point1,point2 = Arm.get_xy_coords()
        tempP2 = point2[:,1:]
        tempP1 = point1[:,1:]
        store1 = distance.cdist(tempB,tempP1,'euclidean')
        store2 = distance.cdist(tempB,tempP2,'euclidean')
        val1 = np.argwhere(store1 == np.min(store1))[0]
        val2 = np.argwhere(store2 == np.min(store2))[0]
        # time = Arm.t[val[0]]
        xB1,yB1 = tempB[val1[0]]
        xB2,yB2 = tempB[val2[0]]
        xA1,yA1 = tempP1[val1[0]]
        xA2,yA2 = tempP2[val2[0]]
        thA1 = Arm.get_theta(xA1,yA1)
        thA2 = Arm.get_theta(xA2,yA2)
        thB1 = Arm.get_theta(xB1,yB1)
        thB2 = Arm.get_theta(xB2,yB2)
        
        diff = thB1-thA1
        dist1 = diff*np.sqrt((xB1-xA1)**2+(yB1-yA1)**2)
        
        diff = thB2-thA2
        dist2 = diff*np.sqrt((xB2-xA2)**2+(yB2-yA2)**2)
        
        f1adj = dist1+f1adj
        f2adj = dist2+f2adj
        return f1adj, f2adj, ballcatch
        
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
    plt.close('all')
    n = 2
    pos1 = -np.pi/4
    pos2 = np.pi/4
    vel = 0
    l1 = (13)*0.0254
    l2 = (12+9)*0.0254
    m1 = 2*5.715264/3
    m2 = 5.715264/3
    damp = 10
    #mech
    mgain = 100
    forceint1 = mgain
    forceint2 = mgain
    maxiter = 20
    ballcatch = False
    #ball
    e = 1 #coefficient of restitution
    mass = 2
    vx = -1.5
    vy = 0
    y = 1
    x = 2
    theta = 0
    ### Time ###
    Ttot = 5 # total time in second
    f_s = 500 # sample frequency (samples/s)
    t_s = np.arange(0,Ttot,1/f_s)
    t_ms = np.arange(0,Ttot*f_s,1)
    
    initial_cond_tpp = [[pos1,pos2],[vel,vel]]
    coeff_tpp = [[l1,l2],[m1,m2],damp]
    initial_cond_ball = [vx,vy,x,y]
    
    ### INIT ###
    Armobj = TpPendulum(n,initial_cond_tpp,coeff_tpp,mgain,t_s,f_s)
    mech = Muscle_Mech(mgain,f_s,t_s)
    Ballobj = Ball(t_s,e,mass,initial_cond_ball)
    
    #### Feedforward ####    
    f1adj = 0
    f2adj = 0
    count = 0
    obstacle = [(0,0)]
    while count < maxiter and not ballcatch:
        Ballobj = Ball(t_s,e,mass,initial_cond_ball)
        Armobj = TpPendulum(n,initial_cond_tpp,coeff_tpp,mgain,t_s,f_s)
        f1adj,f2adj,ballcatch = mech.ff_iterator(Armobj, Ballobj, maxiter, vx,vy, x, y,f1adj,f2adj, ballcatch)
        count += 1
    
    #### Feedback ####
    # xA,yA = Armobj.get_xy()
    # obstacle = (xA[0,2],yA[0,2])
    # for i in range(0,len(t_s)-1):
    #     t = [t_s[i],t_s[i+1]]
    #     dt = abs(t_s[i+1] - t_s[i])        
    #     vx,vy,x,y,ballcatch = Ballobj.simple_ball(vx,vy,x,y,dt,theta,obstacle)
    #     Xp = Armobj.fbsolve(t,x,y,ballcatch)
    #     xA,yA = Armobj.get_xy()
    #     obstacle = (xA[0,2],yA[0,2])
        # Bp = Ballobj.solve(t)
    X = np.stack(Armobj.storeP)
    T = np.stack(Armobj.storeT)
    B = np.stack(Ballobj.storeB)
    Armobj.plot_pendulum_trace()
    Ballobj.plot_ball_trace()
    # anim = Armobj.animate_pendulum()
    anim = Armobj.animate_ball_pendulum(B[:,2:])
        