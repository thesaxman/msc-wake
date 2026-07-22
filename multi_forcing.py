""" A simple approach to visualising effect of n turbines in a single column"""

__author__ = "Ali Alebeedan"
__date__ = "16/7/2026"

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from model_params import wake_params as wp, solver_params as sp
from wake_dynamics import u_point, sinusoid_gamma_t
from unsteady_flow_solver import default_d_dt, solver

from video_utils import save_video


from wake_dynamics import make_turbine

mk = partial(make_turbine, wp)

turbines = [
    mk(0.0),
    mk(5.0, gamma_fn=sinusoid_gamma_t),
    mk(10.0, gamma_deg=-15.0)
]

y0 = (jnp.zeros(sp.nx), jnp.zeros(sp.nx), jnp.zeros(sp.nx))

u1_xt, u2_xt, yc_xt = solver(y0=y0, rhs_func=default_d_dt(turbines, sp), sp=sp).ys

if __name__ == "__main__":
    D = wp.D
    x_grid = sp.x_grid
    y_grid = jnp.linspace(-3.0*D, 3*D, 100)

    # making animation frames
    build_frame = jax.vmap(u_point, in_axes=(0,0,0,None, None), out_axes=1)
    all_frames = jax.vmap(build_frame, in_axes=(None,0,0,None, None), out_axes=0)(x_grid, yc_xt, u1_xt, y_grid, wp)
    assert all_frames.shape == (u1_xt.shape[0], y_grid.size, x_grid.size), all_frames.shape
    import numpy as np
    frames = np.asarray(all_frames)
    u1_xt = np.asarray(u1_xt)
    u2_xt = np.asarray(u2_xt)
    yc_xt = np.asarray(yc_xt)
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, layout= 'constrained', gridspec_kw={'height_ratios': [2.5, 1, 1, 1]})
    ax_field, ax_u1, ax_u2, ax_yc = axes
    
    from matplotlib.ticker import MultipleLocator

    for ax in axes: # introduce gridlines to make figures 
        ax.xaxis.set_major_locator(MultipleLocator(5.0))
        ax.xaxis.set_minor_locator(MultipleLocator(1.0))
        ax.grid(True, which='major', axis='x', lw=0.6, alpha=0.5)
        ax.grid(True, which='minor', axis='x', lw=0.3, alpha=0.25)

    #### Flow field

    mesh = ax_field.pcolormesh(x_grid/D, y_grid/D, frames[0], cmap='plasma',
                        shading='auto', vmin=0, vmax=frames.max())
    fig.colorbar(mesh, ax=ax_field, location = 'top', shrink = 0.5, label = r'$u_1$ [m/s]')
    cl_line, = ax_field.plot(x_grid/D, yc_xt[0]/D, 'w--', lw=1.2, label = '$y_c$')
    ax_field.set_ylabel(r'$y/D$')
    ax_field.legend(loc='upper right')

    ### flow variable profiles

    u1_line, = ax_u1.plot(x_grid/D, u1_xt[0], lw=1.2)
    ax_u1.set_ylim(u1_xt.min(), 1.1*u1_xt.max())
    ax_u1.set_ylabel(r'$u_1$ [m/s]')


    u2_line, = ax_u2.plot(x_grid/D, u2_xt[0], lw=1.2)
    ax_u2.set_ylim(1.1*u2_xt.min(), 1.1*u2_xt.max())
    ax_u2.set_ylabel(r'$u_2$ [m/s]')


    yc_line, = ax_yc.plot(x_grid/D, yc_xt[0], lw=1.2)
    ax_yc.set_ylim(1.1*yc_xt.min(), 1.1*yc_xt.max())
    ax_yc.set_ylabel(r'$y_c$ [m]')
    ax_yc.set_xlabel(r'$x/D$')

    timestamp = ax_field.text(0.02, 0.92, '', transform = ax_field.transAxes, color='w')

    def update(i):
        mesh.set_array(frames[i].ravel())
        cl_line.set_ydata(yc_xt[i]/D)
        u1_line.set_ydata(u1_xt[i])
        u2_line.set_ydata(u2_xt[i])
        yc_line.set_ydata(yc_xt[i])
        timestamp.set_text(rf"t = {sp.ts[i]:.0f} s")
        return mesh, cl_line, u1_line, u2_line, yc_line, timestamp


    ani = FuncAnimation(fig, update, frames=len(sp.ts), interval=50, blit = True)

    save_video(ani,'three_turbine_mf.mp4')