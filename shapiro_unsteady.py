"""
A try at implementing the unsteady PDE to see if Diffrax can take it. 
"""

__author__ = "Ali Alebeedan"
__date__ = "2/7/2026"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from model_params import wake_params
from diffrax import diffeqsolve, Tsit5, ODETerm, SaveAt, PIDController

from wake_dynamics import A, dA_dx, G, u_point
import shapiro_steady as ss
from video_utils import save_video

gamma_rad = jnp.radians(wake_params.gamma_deg)
cos_gamma = jnp.cos(gamma_rad)
sin_gamma = jnp.sin(gamma_rad)

#defining the spatial grid
nx = 800
x = jnp.linspace(wake_params.upstream_bound, wake_params.boundary, nx)
dx = x[1]-x[0]

def expansion(x_grid, p): # compute the expansion term
    return jax.vmap(dA_dx, in_axes=(0,None))(x_grid,p) / A(x_grid,p) 

G_x = G(x,wake_params)                            # precompute the Gaussian forcing term

def delta_u1_0(p):
    return p.UINF*(1.0-jnp.sqrt(1-p.Ct*cos_gamma**2))
S1 = DELTA_U1_0(wake_params)*wake_params.UINF
DELTA_U2_0 = wake_params.UINF*(1.0/4.0*wake_params.Ct*cos_gamma**2*sin_gamma)
S2 = DELTA_U2_0*wake_params.UINF

#solver options
stepsize_controller = PIDController(rtol=1e-5, atol=1e-7, pcoeff=0.3, icoeff=0.3)
t1 = wake_params.t1
nt = 1000
ts = jnp.linspace(0, t1, nt)
max_steps = 10000000

def solver(ts, y0, rhs_func):
    return diffeqsolve(ODETerm(rhs_func), Tsit5(),
                        t0=0, t1=ts[-1], dt0=None, y0=y0,
                        saveat=SaveAt(ts=ts),
                        stepsize_controller=stepsize_controller,
                        max_steps=max_steps)

def d_dx(var, dx): 
    """ First order upwind derivative (flow in +x); zero-gradient inflow BC at index 0. """
    dvar_dx = (var - jnp.roll(var, 1))/dx
    return dvar_dx.at[0].set(0.0)

def make_rhs(p, x_grid):
    
    
    
    def rhs(t, state, args): # system of PDES for u1, u2, yc
        
        u1, u2, yc = state
        
        du1_dt = -p.UINF*d_dx(u1, dx) - p.UINF*expansion*u1 + S1*G(x_grid,p)
        
        du2_dt = -wake_params.UINF*d_dx(u2, dx) - wake_params.UINF*expansion*u2 + S2*G_x
        
        dyc_dt = -wake_params.UINF*d_dx(yc, dx) - u2 # u2 is coupled to the centerline deflection equation
        
        return (du1_dt, du2_dt, dyc_dt)
    
    return rhs

if __name__ == "__main__":

    y0 = (jnp.zeros(nx), jnp.zeros(nx), jnp.zeros(nx))

    u1_xt, u2_xt, yc_xt = solver(ts, y0, rhs).ys



    y = jnp.linspace(-3*wake_params.D, 3*wake_params.D, 100)
    u1_x = jax.vmap(ss.solve_steady_u1(wake_params))(x)
    u2_steady = ss.solve_steady_u2(wake_params)
    u2_x = jax.vmap(u2_steady)(x)
    yc_x = jax.vmap(ss.solve_steady_yc(wake_params, u2_steady))(x)

    # making animation frames
    build_frame = jax.vmap(u_point, in_axes=(0,0,0,None, None), out_axes=1)
    all_frames = jax.vmap(build_frame, in_axes=(None,0,0,None, None), out_axes=0)(x, yc_xt, u1_xt, y, wake_params)
    assert all_frames.shape == (u1_xt.shape[0], y.size, x.size), all_frames.shape
    import numpy as np
    frames = np.asarray(all_frames)
    u1_xt = np.asarray(u1_xt)
    u2_xt = np.asarray(u2_xt)
    yc_xt = np.asarray(yc_xt)

    def separate_wake_video(): # will fix later
        
        raise NotImplementedError("rebuild artists")
        
        fig, ax = plt.subplots(figsize=(10, 4))


        def update(i):
            mesh.set_array(frames[i].ravel())
            line.set_ydata(np.asarray(yc_xt[i])/wake_params.D)
            ax.set_title(rf"$\gamma$ = {wake_params.gamma_deg:.1f}°, t = {ts[i]:.1f} s")
            return mesh, line

        ani = FuncAnimation(fig, update, frames=len(ts), interval=50)
        #ani.save('wake_field_evolution.gif', writer='ffmpeg', dpi=200)

        video_filename = 'wake_field_evolution.mp4'
            
        save_video(ani,video_filename)


    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, layout= 'constrained', gridspec_kw={'height_ratios': [2.5, 1, 1, 1]})
    ax_field, ax_u1, ax_u2, ax_yc = axes

    #### Flow field

    mesh = ax_field.pcolormesh(x/wake_params.D, y/wake_params.D, frames[0], cmap='plasma',
                        shading='auto', vmin=0, vmax=frames.max())
    fig.colorbar(mesh, ax=ax_field, location = 'top', shrink = 0.5, label = r'$u_1$ [m/s]')
    cl_line, = ax_field.plot(x/wake_params.D, yc_xt[0]/wake_params.D, 'w--', lw=1.2, label = 'Unsteady $y_c$')
    ax_field.plot(x/wake_params.D, yc_x/wake_params.D, 'r--', lw=1.0, label='Steady $y_c$')
    ax_field.set_ylabel(r'$y/D$')
    ax_field.legend(loc='upper right')

    ### flow variable profiles

    u1_line_unsteady, = ax_u1.plot(x/wake_params.D, u1_xt[0], lw=1.2)
    ax_u1.plot(x/wake_params.D, u1_x, 'r--', lw=1.0)
    ax_u1.set_ylim(min(u1_xt.min(), u1_x.min()), 1.1*max(u1_xt.max(), u1_x.max()))
    ax_u1.set_ylabel(r'$u_1$ [m/s]')


    u2_line_unsteady, = ax_u2.plot(x/wake_params.D, u2_xt[0], lw=1.2)
    ax_u2.plot(x/wake_params.D, u2_x, 'r--', lw=1.2)
    ax_u2.set_ylim(1.1*min(u2_xt.min(), u2_x.min()), 1.1*max(u2_xt.max(), u2_x.max()))
    ax_u2.set_ylabel(r'$u_2$ [m/s]')


    yc_line_unsteady, = ax_yc.plot(x/wake_params.D, yc_xt[0], lw=1.2)
    ax_yc.plot(x/wake_params.D, yc_x, 'r--', lw=1.2, label='Steady solution')
    ax_yc.set_ylim(1.1*min(yc_xt.min(), yc_x.min()), 1.1*max(yc_xt.max(), yc_x.max()))
    ax_yc.set_ylabel(r'$y_c$ [m]')
    ax_yc.set_xlabel(r'$x/D$')

    timestamp = ax_field.text(0.02, 0.92, '', transform = ax_field.transAxes, color='w')

    def update(i):
        mesh.set_array(frames[i].ravel())
        cl_line.set_ydata(yc_xt[i]/wake_params.D)
        u1_line_unsteady.set_ydata(u1_xt[i])
        u2_line_unsteady.set_ydata(u2_xt[i])
        yc_line_unsteady.set_ydata(yc_xt[i])
        timestamp.set_text(rf"$\gamma$ = {wake_params.gamma_deg:.1f}°, t = {ts[i]:.0f} s")
        return mesh, cl_line, u1_line_unsteady, u2_line_unsteady, yc_line_unsteady, timestamp


    ani = FuncAnimation(fig, update, frames=len(ts), interval=50, blit = True)

    save_video(ani,'full_wake_evolution.mp4')
    #plt.show()