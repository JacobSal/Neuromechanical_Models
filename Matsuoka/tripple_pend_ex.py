# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 11:25:26 2021

@author: jsalm
"""
import matplotlib.pyplot as plt
import numpy as np

from sympy import symbols
from sympy.physics import mechanics

from sympy import Dummy, lambdify
from scipy.integrate import odeint

#animation functions
from matplotlib import animation
from IPython.display import HTML

def integrate_pendulum(n, times, exforce = [],
                       initial_positions=135,
                       initial_velocities=0,
                       lengths=None, masses=1, dampening=0):
    """Integrate a multi-pendulum with `n` sections"""
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
    f = mechanics.dynamicsymbols('f:{0}'.format(n))
    
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
        forces.append((Pi, f[i]*A.y+m[i]*g*A.x))
        kinetic_odes.append(q[i].diff(t) - u[i])

        P = Pi

    # Generate equations of motion
    KM = mechanics.KanesMethod(A, q_ind=q, u_ind=u,
                               kd_eqs=kinetic_odes)
    fr, fr_star = KM.kanes_equations(particles, forces)
    
    #-----------------------------------------------------
    # Step 3: numerically evaluate equations and integrate

    # initial positions and velocities – assumed to be given in degrees
    y0 = np.deg2rad(np.concatenate([np.broadcast_to(initial_positions, n),
                                    np.broadcast_to(initial_velocities, n)]))
        
    # lengths and masses
    if lengths is None:
        lengths = np.ones(n) / n
    lengths = np.broadcast_to(lengths, n)
    masses = np.broadcast_to(masses, n)
    damp = np.broadcast_to(dampening,n)

    # Fixed parameters: gravitational , lengths, and masses
    parameters = [g] + list(l) + list(m) + list(k)
    parameter_vals = [9.81] + list(lengths) + list(masses) + list(damp)

    # define symbols for unknown parameters
    dynamic = q + u + f
    unknowns = [Dummy() for i in dynamic]
    unknown_dict = dict(zip(dynamic, unknowns))
    kds = KM.kindiffdict()

    # substitute unknown symbols for qdot terms
    mm_sym = KM.mass_matrix_full.subs(kds).subs(unknown_dict)
    fo_sym = KM.forcing_full.subs(kds).subs(unknown_dict)

    # create functions for numerical calculation 
    mm_func = lambdify(unknowns + parameters, mm_sym)
    fo_func = lambdify(unknowns + parameters, fo_sym)

    # function which computes the derivatives of parameters
    def gradient(y, t, args):
        vals = np.concatenate((y, args))
        sol = np.linalg.solve(mm_func(*vals), fo_func(*vals))
        return np.array(sol).T[0]

    # ODE integration
    return odeint(gradient, y0, times, args=(parameter_vals,))
    

def get_xy_coords(p, lengths=None):
    """Get (x, y) coordinates from generalized coordinates p"""
    p = np.atleast_2d(p)
    n = p.shape[1] // 2
    if lengths is None:
        lengths = np.ones(n) / n
    zeros = np.zeros(p.shape[0])[:, None]
    x = np.hstack([zeros, lengths * np.sin(p[:, :n])])
    y = np.hstack([zeros, -lengths * np.cos(p[:, :n])])
    return np.cumsum(x, 1), np.cumsum(y, 1)

def get_angles(p, lengths = None):
    pass

def plot_pendulum_trace(p):
    x, y = get_xy_coords(p)
    plt.figure("tripple Pendulum Trace")
    plt.plot(x, y);
    plt.xlabel("position (m)")
    plt.ylabel("position (m)")
    plt.show()
    # plt.close()
    return 0

def set_new_tpp(Xp,n_p):
    n = Xp.shape[1] // 2
    n_p_new = n_p.copy()
    n_p_new[0] = list(Xp[-1,:n])
    n_p_new[1] = list(Xp[-1,n:])
    return n_p_new
    
def animate_pendulum(p,t):
    x, y = get_xy_coords(p)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.set(xlim=(-1, 1), ylim=(-1, 1))

    line, = ax.plot([], [], 'o-', lw=2)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        line.set_data(x[i], y[i])
        return line,

    anim = animation.FuncAnimation(fig, animate, frames=len(t),
                                   interval=1000 * t.max() / len(t),
                                   blit=True, init_func=init)
    # plt.close(fig)
    return anim

if __name__ == '__main__':
    Ttot = 1
    # total time in second
    f_s = 50 # sample frequency (samples/s)
    t = np.arange(0,Ttot,1/f_s)
    n = 3
    jj = []
    n_p = [[135,135,135],[0,0,0],[1,1,1],[1,1,1],1]
    exforce = [0,0,0]
    p = integrate_pendulum(n,t,exforce,n_p[0],n_p[1],n_p[2],n_p[3],n_p[4])
    
    # for i in range(len(t)-1):
    #     t_s = [t[i],t[i+1]]
    #     print(n_p[0])
    #     p = integrate_pendulum(n,t_s,exforce,n_p[0],n_p[1],n_p[2],n_p[3],n_p[4])
    #     n_p = set_new_tpp(p,n_p)
    #     jj.append(p[-1,:])
    # p = np.stack(jj)
    x, y = get_xy_coords(p)    
    plt.figure("tripple Pendulum Trace")
    plt.plot(x, y);
    plt.waitforbuttonpress(5)
    plt.close()
    anim = animate_pendulum(p,t)
    # HTML(anim.to_html5_video())
    # HTML('<video controls loop src="http://jakevdp.github.io/videos/triple-pendulum.mp4" />')