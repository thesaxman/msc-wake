"""_summary_ This script is an attempt at adding time-dependence in forcing.
"""
    
__author__ = "Ali Alebeedan"
__date__ = "10/7/2026"

from matplotlib.animation import FuncAnimation
from dataclasses import replace

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from model_params import wake_params as wp, solver_params as sp
from wake_dynamics import G, u_point, sinusoid_gamma_t
from unsteady_flow_solver import default_d_dt, solver
from video_utils import save_video



wp_sinusoid = replace(wp, gamma_fn = sinusoid_gamma_t)




if __name__ == "__main__":
    import shapiro_steady as ss
    u1_x = jax.vmap(ss.solve_steady_u1(wp_sinusoid))(sp.x_grid)
    u2_steady = ss.solve_steady_u2(wp_sinusoid)
    u2_x = jax.vmap(u2_steady)(sp.x_grid)
    yc_x = jax.vmap(ss.solve_steady_yc(wp_sinusoid, u2_steady))(sp.x_grid)
    y0 = (u1_x, u2_x, yc_x)

    u1_xt, u2_xt, yc_xt = solver(y0, default_d_dt(wp_sinusoid, sp), sp).ys
    y_grid = jnp.linspace(-3*wp_sinusoid.D, 3*wp_sinusoid.D, 100)
    
    # making animation frames
    build_frame = jax.vmap(u_point, in_axes=(0,0,0,None, None), out_axes=1)
    all_frames = jax.vmap(build_frame, in_axes=(None,0,0,None, None), out_axes=0)(sp.x_grid, yc_xt, u1_xt, y_grid, wp_sinusoid)
    assert all_frames.shape == (u1_xt.shape[0], y_grid.size, sp.x_grid.size), all_frames.shape
    import numpy as np
    frames = np.asarray(all_frames)
    u1_xt = np.asarray(u1_xt)
    u2_xt = np.asarray(u2_xt)
    yc_xt = np.asarray(yc_xt)
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, layout= 'constrained', gridspec_kw={'height_ratios': [2.5, 1, 1, 1]})
    ax_field, ax_u1, ax_u2, ax_yc = axes
    

    #### Flow field

    mesh = ax_field.pcolormesh(sp.x_grid/wp_sinusoid.D, y_grid/wp_sinusoid.D, frames[0], cmap='plasma',
                        shading='auto', vmin=0, vmax=frames.max())
    fig.colorbar(mesh, ax=ax_field, location = 'top', shrink = 0.5, label = r'$u_1$ [m/s]')
    cl_line, = ax_field.plot(sp.x_grid/wp_sinusoid.D, yc_xt[0]/wp_sinusoid.D, 'w--', lw=1.2, label = '$y_c$')
    ax_field.set_ylabel(r'$y/D$')
    ax_field.legend(loc='upper right')

    ### flow variable profiles

    u1_line, = ax_u1.plot(sp.x_grid/wp_sinusoid.D, u1_xt[0], lw=1.2)
    ax_u1.set_ylim(u1_xt.min(), 1.1*u1_xt.max())
    ax_u1.set_ylabel(r'$u_1$ [m/s]')


    u2_line, = ax_u2.plot(sp.x_grid/wp_sinusoid.D, u2_xt[0], lw=1.2)
    ax_u2.set_ylim(1.1*u2_xt.min(), 1.1*u2_xt.max())
    ax_u2.set_ylabel(r'$u_2$ [m/s]')


    yc_line, = ax_yc.plot(sp.x_grid/wp_sinusoid.D, yc_xt[0], lw=1.2)
    ax_yc.set_ylim(1.1*yc_xt.min(), 1.1*yc_xt.max())
    ax_yc.set_ylabel(r'$y_c$ [m]')
    ax_yc.set_xlabel(r'$x/D$')

    timestamp = ax_field.text(0.02, 0.92, '', transform = ax_field.transAxes, color='w')

    def update(i):
        mesh.set_array(frames[i].ravel())
        cl_line.set_ydata(yc_xt[i]/wp_sinusoid.D)
        u1_line.set_ydata(u1_xt[i])
        u2_line.set_ydata(u2_xt[i])
        yc_line.set_ydata(yc_xt[i])
        timestamp.set_text(rf"$\gamma$ = {jnp.degrees(wp_sinusoid.gamma_at(sp.ts[i])):.1f}°, t = {sp.ts[i]:.0f} s")
        return mesh, cl_line, u1_line, u2_line, yc_line, timestamp


    ani = FuncAnimation(fig, update, frames=len(sp.ts), interval=50, blit = True)

    save_video(ani,'sinusoidal_yaw.mp4')