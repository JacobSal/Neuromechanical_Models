# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 22:11:09 2021

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

lengths = 1
masses = 1
dampening = 1
n = 3
initial_positions = [0.2,0.2,0.2]
initial_velocities = [0,0,0]

# initial_positions = [0.200198, 0.2, .2]
# initial_velocities = [0.0197846, 4.04408e-05, 1.71495e-08]

# initial_positions = [0.00349066,	0.00349066,	0.00349066]	
# initial_velocities = [0, 0,	0]


force = 3

Ttot = 1
# total time in second
f_s = 50 # sample frequency (samples/s)
t_in = np.arange(0,Ttot,1/f_s)
# times = t
# times = np.array([t[1],t[2]])
#-------------------------------------------------
# Step 1: construct the pendulum model

# Generalized coordinates and velocities
# (in this case, angular positions & velocities of each mass) 
q = mechanics.dynamicsymbols('q:{0}'.format(n))
u = mechanics.dynamicsymbols('u:{0}'.format(n))
f = mechanics.dynamicsymbols('f:{0}'.format(n))

# mass and length and dampening
m = symbols('m:{0}'.format(n))
l = symbols('l:{0}'.format(n))
k = symbols('k:{0}'.format(n))

# gravity and time symbols
g, t = symbols('g,t')

#force

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

y0 = np.concatenate([np.broadcast_to(initial_positions, n),
                                   np.broadcast_to(initial_velocities, n)])
       
# lengths and masses
if lengths is None:
    lengths = np.ones(n) / n
lengths = np.broadcast_to(lengths, n)
masses = np.broadcast_to(masses, n)
damp = np.broadcast_to(dampening,n)
exforces = np.broadcast_to(force,n)

# Fixed parameters: gravitational , lengths, and masses
parameters = [g] + list(l) + list(m) + list(k) + list(f)
parameter_vals = [9.81] + list(lengths) + list(masses) + list(damp) + list(exforces)

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

# function which computes the derivatives of parameters
def gradient(y, t, args):
    vals = np.concatenate((y, args))
    sol = np.linalg.solve(mm_func(*vals), fo_func(*vals))
    return np.array(sol).T[0]

jj = []
jj.append(y0)
for i in range(0,len(t_in)-1):
    times = [t_in[i],t_in[i+1]]
    xp = odeint(gradient, y0, times, args=(parameter_vals,))
    jj.append(xp[-1,:])
    y0 = xp[-1,:]
    