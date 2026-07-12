"""
This script generates the steady state flow field introduced in the tutorial d-
ocument, and plots streamwise, lateral velocity and wake centreline deflection
against the distance downstream of the turbine.
"""

__author__ = "Ali Alebeedan"
__date__ = "5/6/2026"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from model_params import *
from diffrax import diffeqsolve, Tsit5, ODETerm, SaveAt, PIDController

def wake_expansion(x, D, kw):
    return 1+kw*jnp.log(1+jnp.exp(2*x/D))

dw = lambda x: wake_expansion(x, D, kw)
dw0 = dw(0)
u1 =lambda x: UINF*(1-jnp.sqrt(1-Ct*jnp.cos(jnp.radians(gamma))**2))  * (dw0/dw(x))**2
u2 = lambda x: UINF/4*Ct* jnp.cos(jnp.radians(gamma))**2 * jnp.sin(jnp.radians(gamma)) * (dw0/dw(x))**2

def generate_wake_deflection(x, Ct, gamma, t1):
    
    gamma = jnp.radians(gamma)
    def dy_dx(t, y, args):
        return -1/4*Ct* jnp.cos(gamma) **2 * jnp.sin(gamma) * (dw0/dw(t))**2
    
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

dw0 = dw(0) 
dw_x = dw(x)                                                      # (200,)
u1_x = UINF*(1 - jnp.sqrt(1 - Ct*jnp.cos(jnp.radians(gamma))**2)) * (dw0/dw_x)**2

# --- broadcast to 2D field ---
# X, Y are (150, 200); yc, dw_x, u1_x are (200,) — broadcast along y axis
sigma_x = sigma0 * dw_x                          # (200,) local wake width
     
A_x     = A(x)                                    # A(x) — set per your model
                               # dw(0) — set per your model
gaussian = (1.0 / (jnp.sqrt(2*jnp.pi) * sigma_x)) * \
            jnp.exp(-0.5 * ((Y - yc) / sigma_x)**2)   # (150, 200)

u_field = u1_x * A_x * gaussian                  # (150, 200)

# --- plot ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# velocity deficit field
cf = axes[0].pcolormesh(X/D, Y/D, u_field, cmap='RdBu_r', shading='auto')
axes[0].plot(x/D, yc/D, color='white', linestyle='dashed', linewidth=1.5,
             label='Wake centreline $y_c(x)$')
plt.colorbar(cf, ax=axes[0], label='$u(x,y)$ [m/s]')
axes[0].set_ylabel('$y/D$')
axes[0].set_title('Streamwise velocity deficit')
axes[0].legend()

# centreline trajectory alone
axes[1].plot(x/D, yc/D, color='steelblue', label='$y_c(x)$')
axes[1].set_xlabel('$x/D$')
axes[1].set_ylabel('$y_c/D$')
axes[1].set_title('Wake centreline deflection')
axes[1].legend()

plt.tight_layout()
plt.savefig("wake_field_2d.png", dpi=150)
plt.show()