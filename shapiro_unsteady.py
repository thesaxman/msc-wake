"""
A try at implementing the unsteady PDE to see if Diffrax can take it. 
"""

__author__ = "Ali Alebeedan"
__date__ = "2/7/2026"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from model_params import wake_params as wp, solver_params as sp
from diffrax import diffeqsolve, Tsit5, ODETerm, SaveAt, PIDController

from wake_dynamics import A, dA_dx, G, u_point, WakeParams, expansion
import unsteady_flow_solver as flowsolve
import shapiro_steady as ss
from video_utils import save_video


if __name__ == "__main__":

    x_grid = sp.x_grid

    y0 = (jnp.zeros(sp.nx), 
          jnp.zeros(sp.nx), 
          jnp.zeros(sp.nx))

    u1_xt, u2_xt, yc_xt = flowsolve.solver(y0, flowsolve.default_d_dt(wp, sp), sp).ys



    y_grid = jnp.linspace(-3*wp.D, 3*wp.D, 100)
    u1_x = jax.vmap(ss.solve_steady_u1(wp))(x_grid)
    u2_steady = ss.solve_steady_u2(wp)
    u2_x = jax.vmap(u2_steady)(x_grid)
    yc_x = jax.vmap(ss.solve_steady_yc(wp, u2_steady))(x_grid)

    # making animation frames
    build_frame = jax.vmap(u_point, in_axes=(0,0,0,None, None), out_axes=1)
    all_frames = jax.vmap(build_frame, in_axes=(None,0,0,None, None), out_axes=0)(x_grid, yc_xt, u1_xt, y_grid, wp)
    assert all_frames.shape == (u1_xt.shape[0], y_grid.size, x_grid.size), all_frames.shape
    import numpy as np
    frames = np.asarray(all_frames)
    u1_xt = np.asarray(u1_xt)
    u2_xt = np.asarray(u2_xt)
    yc_xt = np.asarray(yc_xt)

    def separate_wake_video():
        
        x_norm = x_grid/wp.D
        y_norm = y_grid/wp.D
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        mesh = ax.pcolormesh(x_norm, y_norm, frames[0], cmap = 'plasma', shading='auto', vmin = 0, vmax = frames.max())
        fig.colorbar(mesh, ax = ax, location = 'right', label = r'$u_1$ [m/s]')
        cl_line, = ax.plot(x_norm, yc_xt[0]/wp.D, 'w--', lw=1.2, label = 'Unsteady $y_c$')
        ax.plot(x_grid/wp.D, yc_x/wp.D, 'r--', lw=1.0, label='Steady $y_c$')
        ax.set_ylabel(r'$y/D$')
        ax.legend(loc='upper right')
        
        timestamp = ax.text(0.02, 0.92, '', transform = ax.transAxes, color='w')
        
        def update(i):
            mesh.set_array(frames[i].ravel())
            cl_line.set_ydata(yc_xt[i]/wp.D)
            timestamp.set_text(rf"$\gamma$ = {wp.gamma_deg:.1f}°, t = {sp.ts[i]:.0f} s")
            return mesh, cl_line, timestamp
        
        ani = FuncAnimation(fig, update, frames=len(sp.ts), interval=50)
        video_filename = 'wake_field_evolution.mp4'
        save_video(ani,video_filename)
    
    def wake_profiles():
        
        x_norm = x_grid/wp.D
        y_norm = y_grid/wp.D
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, layout= 'constrained')
        ax_u1, ax_u2, ax_yc = axes
        
        ### flow variable profiles

        u1_line_unsteady, = ax_u1.plot(x_norm, u1_xt[0], lw=1.2)
        ax_u1.plot(x_norm, u1_x, 'r--', lw=1.0)
        ax_u1.set_ylim(min(u1_xt.min(), u1_x.min()), 1.1*max(u1_xt.max(), u1_x.max()))
        ax_u1.set_ylabel(r'$u_1$ [m/s]')

        u2_line_unsteady, = ax_u2.plot(x_norm, u2_xt[0], lw=1.2)
        ax_u2.plot(x_norm, u2_x, 'r--', lw=1.2)
        ax_u2.set_ylim(1.1*min(u2_xt.min(), u2_x.min()), 1.1*max(u2_xt.max(), u2_x.max()))
        ax_u2.set_ylabel(r'$u_2$ [m/s]')

        yc_line_unsteady, = ax_yc.plot(x_norm, yc_xt[0], lw=1.2)
        ax_yc.plot(x_norm, yc_x, 'r--', lw=1.2, label='Steady solution')
        ax_yc.set_ylim(1.1*min(yc_xt.min(), yc_x.min()), 1.1*max(yc_xt.max(), yc_x.max()))
        ax_yc.set_ylabel(r'$y_c$ [m]')
        ax_yc.set_xlabel(r'$x/D$')

        timestamp = ax_u1.set_title('')

        def update(i):
            u1_line_unsteady.set_ydata(u1_xt[i])
            u2_line_unsteady.set_ydata(u2_xt[i])
            yc_line_unsteady.set_ydata(yc_xt[i])
            timestamp.set_text(rf"$\gamma$ = {wp.gamma_deg:.1f}°, t = {sp.ts[i]:.0f} s")
            return u1_line_unsteady, u2_line_unsteady, yc_line_unsteady, timestamp


        ani = FuncAnimation(fig, update, frames=len(sp.ts), interval=50, blit = True)

        save_video(ani,'wake_profiles_evolution.mp4')

    def full_wake():
        
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, layout= 'constrained', gridspec_kw={'height_ratios': [2.5, 1, 1, 1]})
        ax_field, ax_u1, ax_u2, ax_yc = axes

        #### Flow field

        mesh = ax_field.pcolormesh(x_grid/wp.D, y_grid/wp.D, frames[0], cmap='plasma',
                            shading='auto', vmin=0, vmax=frames.max())
        fig.colorbar(mesh, ax=ax_field, location = 'top', shrink = 0.5, label = r'$u_1$ [m/s]')
        cl_line, = ax_field.plot(x_grid/wp.D, yc_xt[0]/wp.D, 'w--', lw=1.2, label = 'Unsteady $y_c$')
        ax_field.plot(x_grid/wp.D, yc_x/wp.D, 'r--', lw=1.0, label='Steady $y_c$')
        ax_field.set_ylabel(r'$y/D$')
        ax_field.legend(loc='upper right')

        ### flow variable profiles

        u1_line_unsteady, = ax_u1.plot(x_grid/wp.D, u1_xt[0], lw=1.2)
        ax_u1.plot(x_grid/wp.D, u1_x, 'r--', lw=1.0)
        ax_u1.set_ylim(min(u1_xt.min(), u1_x.min()), 1.1*max(u1_xt.max(), u1_x.max()))
        ax_u1.set_ylabel(r'$u_1$ [m/s]')


        u2_line_unsteady, = ax_u2.plot(x_grid/wp.D, u2_xt[0], lw=1.2)
        ax_u2.plot(x_grid/wp.D, u2_x, 'r--', lw=1.2)
        ax_u2.set_ylim(1.1*min(u2_xt.min(), u2_x.min()), 1.1*max(u2_xt.max(), u2_x.max()))
        ax_u2.set_ylabel(r'$u_2$ [m/s]')


        yc_line_unsteady, = ax_yc.plot(x_grid/wp.D, yc_xt[0], lw=1.2)
        ax_yc.plot(x_grid/wp.D, yc_x, 'r--', lw=1.2, label='Steady solution')
        ax_yc.set_ylim(1.1*min(yc_xt.min(), yc_x.min()), 1.1*max(yc_xt.max(), yc_x.max()))
        ax_yc.set_ylabel(r'$y_c$ [m]')
        ax_yc.set_xlabel(r'$x/D$')

        timestamp = ax_field.text(0.02, 0.92, '', transform = ax_field.transAxes, color='w')

        def update(i):
            mesh.set_array(frames[i].ravel())
            cl_line.set_ydata(yc_xt[i]/wp.D)
            u1_line_unsteady.set_ydata(u1_xt[i])
            u2_line_unsteady.set_ydata(u2_xt[i])
            yc_line_unsteady.set_ydata(yc_xt[i])
            timestamp.set_text(rf"$\gamma$ = {wp.gamma_deg:.1f}°, t = {sp.ts[i]:.0f} s")
            return mesh, cl_line, u1_line_unsteady, u2_line_unsteady, yc_line_unsteady, timestamp


        ani = FuncAnimation(fig, update, frames=len(sp.ts), interval=50, blit = True)

        save_video(ani,'full_wake_evolution.mp4')
    
    