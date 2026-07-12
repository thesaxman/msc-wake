"""
This script generates a video of the perturbed steady-state solution.
"""

__author__ = "Ali Alebeedan"
__date__ = "1/7/2026"

import jax # to use jax.vmap for vectorization
import jax.numpy as jnp
import matplotlib.pyplot as plt
from model_params import *
from diffrax import diffeqsolve, Tsit5, ODETerm, SaveAt, PIDController

def wake_expansion(x, D, kw):
    return 1+kw*jnp.log(1+jnp.exp(2*x/D))

dw = lambda x: wake_expansion(x, D, kw)
u1 =lambda x: UINF*(1-jnp.sqrt(1-Ct*jnp.cos(jnp.radians(gamma))**2))  * (dw(0)/dw(x))**2
u2 = lambda x: UINF/4*Ct* jnp.cos(jnp.radians(gamma))**2 * jnp.sin(jnp.radians(gamma)) * (dw(0)/dw(x))**2

def generate_wake_deflection(x, Ct, gamma, t1):
    
    gamma = jnp.radians(gamma)
    def dy_dx(t, y, args):
        return -1/4*Ct* jnp.cos(gamma) **2 * jnp.sin(gamma) * (dw(0)/dw(t))**2
    
    term = ODETerm(dy_dx)
    solver = Tsit5()
    saveat = SaveAt(dense=True)
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
    sol = diffeqsolve(term, solver, t0=0, t1=t1, dt0=None, y0=0., saveat=saveat, stepsize_controller=stepsize_controller)
                                                                           
    return jax.vmap(sol.evaluate)(x)

x = jnp.linspace(0, boundary, 100)
y = jnp.linspace(-3*D, 3*D, 100)
X, Y = jnp.meshgrid(x, y)
yc = generate_wake_deflection(x, Ct, gamma, t1 = float(x[-1]))
u1 = u1(x)
u2 = u2(x)
A = lambda x: 0.25 * jnp.pi * D ** 2 * dw(x) ** 2
gaussian = 1/(jnp.sqrt(2*jnp.pi)*sigma0*dw(x)) * jnp.exp(-0.5 * (Y - yc[:, None])**2 / (sigma0*dw(x))**2)
u_field = u1[:, None] * ( gaussian * A(x))