"""
A try at implementing the unsteady PDE to see if Diffrax can take it. 
"""

__author__ = "Ali Alebeedan"
__date__ = "2/7/2026"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
from matplotlib.animation import FuncAnimation
from model_params import *
from diffrax import diffeqsolve, Tsit5, ODETerm, SaveAt, PIDController
from shapiro_steady import yc_x1, x as steady_x, u1_x, u2_x

#defining the wake expansion function and its derivative

def wake_expansion(x, D, kw): # General wake expansion function
    return 1+kw*jnp.log(1+jnp.exp(2*(x-D)/D))
def dw(x): # Wake expansion with flow parameters specified
    return wake_expansion(x, D, kw)
def A(x): # Wake area function
    return jnp.pi * D ** 2 / 4 * dw(x) ** 2
dA_dx = jax.grad(A)
gamma_rad = jnp.radians(gamma)
cos_gamma = jnp.cos(gamma_rad)
sin_gamma = jnp.sin(gamma_rad)
def G(x): # Gaussian forcing function replacing Dirac delta function
    return 1/(jnp.sqrt(2*jnp.pi)*D/2) * jnp.exp(-0.5 * (x)**2 / (D/2)**2)

#defining the spatial grid
nx = 800
x = jnp.linspace(-4.0*D, float(boundary), nx)
dx = x[1]-x[0]

expansion = jax.vmap(dA_dx)(x) / A(x) # precompute the expansion term
G_x = G(x)                            # precompute the Gaussian forcing term

DELTA_U1_0 = UINF*(1.0-jnp.sqrt(1-Ct*cos_gamma**2))
S1 = DELTA_U1_0*UINF
DELTA_U2_0 = UINF*(1.0/4.0*Ct*cos_gamma**2*sin_gamma)
S2 = DELTA_U2_0*UINF

#solver options
stepsize_controller = PIDController(rtol=1e-5, atol=1e-7, pcoeff=0.3, icoeff=0.3)
t1 = 3000.0
nt = 1000
ts = jnp.linspace(0, t1, nt)
max_steps = 10000000

def solver(ts, y0, rhs_func):
    return diffeqsolve(ODETerm(rhs_func), Tsit5(),
                        t0=0, t1=ts[-1], dt0=None, y0=y0,
                        saveat=SaveAt(ts=ts),
                        stepsize_controller=stepsize_controller,
                        max_steps=max_steps)



def rhs(t, state, args): # system of PDES for u1, u2, yc
    
    u1, u2, yc = state
    
    du1_dx = (u1 - jnp.roll(u1, 1)) / dx # Upwind scheme for spatial derivative
    du1_dx = du1_dx.at[0].set(0.0) # Boundary condition at x=-4D
    du1_dt = -UINF*du1_dx - UINF*expansion*u1 + S1*G_x
    
    du2_dx = (u2 - jnp.roll(u2, 1)) / dx # Upwind scheme for spatial derivative
    du2_dx = du2_dx.at[0].set(0.0) # Boundary condition at x=-4D
    du2_dt = -UINF*du2_dx - UINF*expansion*u2 + S2*G_x
    
    dyc_dx = (yc - jnp.roll(yc, 1)) / dx
    dyc_dx = dyc_dx.at[0].set(0.0)
    dyc_dt = -UINF*dyc_dx - u2 # u2 is coupled to the centerline deflection equation
    
    return (du1_dt, du2_dt, dyc_dt)

y0 = (jnp.zeros(nx), jnp.zeros(nx), jnp.zeros(nx))

u1_xt, u2_xt, yc_xt = solver(ts, y0, rhs).ys



#Flow field can be first evaluated along x

y = jnp.linspace(-3*D, 3*D, 100)

# Wake expansion effects onto 2D
def sigma(x):
    return sigma0 * dw(x)
def gaussian(x, yc, y):
    return 0.5 * (D / 2 / sigma0)**2 * \
           jnp.exp(-0.5 * ((y - yc) / sigma(x))**2)
#gaussian_2d = jax.vmap(gaussian, in_axes=(0,0, None), out_axes=1)(x, yc_x, y)
def u1_point(x, y, u1, yc):
    return u1 * gaussian(x, yc, y)

build_frame = jax.vmap(u1_point, in_axes=(0,None,0,0), out_axes=1)
all_frames = jax.vmap(build_frame, in_axes=(None,None,0,0), out_axes=0)(x, y, u1_xt, yc_xt)
assert all_frames.shape == (u1_xt.shape[0], y.size, x.size), all_frames.shape
import numpy as np
frames = np.asarray(all_frames)

fig, ax = plt.subplots(figsize=(10, 4))
mesh = ax.pcolormesh(x/D, y/D, frames[0], cmap='RdBu_r',
                     shading='auto', vmin=frames.min(), vmax=frames.max())
line, = ax.plot(x/D, np.asarray(yc_xt[0])/D, 'w--', lw=1.2)
stead_line, = ax.plot(steady_x/D, yc_x1/D, 'r--', lw=1.2, label='Steady solution')

def update(i):
    mesh.set_array(frames[i].ravel())
    line.set_ydata(np.asarray(yc_xt[i])/D)
    ax.set_title(f"$\gamma$ = {gamma:.1f}°, t = {ts[i]:.1f} s")
    return mesh, line

ani = FuncAnimation(fig, update, frames=len(ts[ts <= 600]), interval=50)
#ani.save('wake_field_evolution.gif', writer='ffmpeg', dpi=200)

video_filename = 'wake_field_evolution.mp4'

def save_video():
    ani.save(video_filename, writer='ffmpeg', dpi=200, fps=20)
       
save_video()


fig, axes = plt.subplots(3, 1, figsize=(10, 4))

u1_line_unsteady, = axes[0].plot(x/D, jnp.asarray(u1_xt[0]), lw=1.2)
u1_line_steady, = axes[0].plot(steady_x/D, u1_x, 'r--', lw=1.2, label='Steady solution')
axes[0].set_ylim(min(u1_xt.min(), u1_x.min()), 1.1*max(u1_xt.max(), u1_x.max()))
axes[0].set_ylabel('$u_1$ [m/s]')
axes[0].legend()

u2_line_unsteady, = axes[1].plot(x/D, jnp.asarray(u2_xt[0]), lw=1.2)
u2_line_steady, = axes[1].plot(steady_x/D, u2_x, 'r--', lw=1.2, label='Steady solution')
axes[1].set_ylim(min(u2_xt.min(), u2_x.min()), 1.1*max(u2_xt.max(), u2_x.max()))
axes[1].set_ylabel('$u_2$ [m/s]')
axes[1].legend()

yc_line_unsteady, = axes[2].plot(x/D, jnp.asarray(yc_xt[0]), lw=1.2)
yc_line_steady, = axes[2].plot(steady_x/D, yc_x1, 'r--', lw=1.2, label='Steady solution')
axes[2].set_ylim(min(yc_xt.min(), yc_x1.min()), 1.1*max(yc_xt.max(), yc_x1.max()))
axes[2].set_ylabel('$y_c$ [m]')
axes[2].legend()


def update(i):
    u1_line_unsteady.set_ydata(jnp.asarray(u1_xt[i]))
    u2_line_unsteady.set_ydata(jnp.asarray(u2_xt[i]))
    yc_line_unsteady.set_ydata(jnp.asarray(yc_xt[i]))
    axes[0].set_title(f"$\gamma$ = {gamma:.1f}°, t = {ts[i]:.1f} s")
    return u1_line_unsteady
axes[-1].set_xlabel('$x/D$')


ani = FuncAnimation(fig, update, frames=len(ts), interval=50)

video_filename = 'wake_variables_evolution.mp4'

save_video()
#plt.show()