"""_summary_ This script is an attempt at adding time-dependence in forcing.
"""
    
__author__ = "Ali Alebeedan"
__date__ = "10/7/2026"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from model_params import wake_params
from wake_dynamics import G, u_point

from video_utils import save_video


gamma0_rad = jnp.radians(wake_params.gamma_deg)
gamma_amplitude_rad = jnp.radians(30)
gamma_frequency = 0.01827665508523  # Frequency of the sinusoidal variation in Hz

def sinusoidal_variation(t, x0, amplitude, frequency):
    """
    Function to output a sinusoidally time-varying quantity x about mean x = 0.
    
    Parameters:
    t : float
        Time variable/parameter.
    x0 : float
        Initial value.
    amplitude : float
        Amplitude of the variation.
    frequency : float
        Frequency of the variation in Hz.
        
    Returns:
    float
        Time-varying value.
    """
    offset = jnp.arcsin(x0 / amplitude) / (2 * jnp.pi * frequency)
    return amplitude * jnp.sin(2 * jnp.pi * frequency * (t + offset))

def gamma_t(t, gamma0):
    """Time varying function of the yaw angle.

    Args:
        t (Float): time variable
        gamma0 (Float): initial yaw angle
    """
    
    return sinusoidal_variation(t, gamma0, amplitude=gamma_amplitude_rad, frequency=gamma_frequency)

def S1(t,p):
    return p.UINF*p.UINF*(1.0-jnp.sqrt(1-p.Ct*jnp.cos(gamma_t(t,p.gamma))))

def S2(t,p):
    return wake_params.UINF*( 1.0/4.0 * wake_params.Ct * jnp.cos(gamma_t(t,p.gamma))**2 * jnp.sin(gamma_t(t,p.gamma)))


from shapiro_unsteady import solver, d_dx, expansion, ts, nx, x

def make_rhs(p, x_grid):
    
    G_x = G(x_grid, p)
    expansion_x = expansion(x_grid, p)
    dx = x_grid[1] - x_grid[0]
    
    def rhs(t, state, args):
    
        u1, u2, yc = state
    
        du1_dt = -p.UINF * d_dx(u1, dx) - p.UINF * expansion_x * u1 + S1(t, p)*G_x
        du2_dt = -p.UINF * d_dx(u2, dx) - p.UINF * expansion_x * u2 + S2(t, p)*G_x
        dyc_dt = -p.UINF * d_dx(yc, dx) - u2
        
        return (du1_dt, du2_dt, dyc_dt)
    
    return rhs


if __name__ == "__main__":
    
    y0 = (jnp.zeros(nx), jnp.zeros(nx), jnp.zeros(nx))

    u1_xt, u2_xt, yc_xt = solver(ts, y0, make_rhs(wake_params, x)).ys
    y = jnp.linspace(-3*wake_params.D, 3*wake_params.D, 100)
    
    # making animation frames
    build_frame = jax.vmap(u_point, in_axes=(0,0,0,None, None), out_axes=1)
    all_frames = jax.vmap(build_frame, in_axes=(None,0,0,None, None), out_axes=0)(x, yc_xt, u1_xt, y, wake_params)
    assert all_frames.shape == (u1_xt.shape[0], y.size, x.size), all_frames.shape
    import numpy as np
    frames = np.asarray(all_frames)
    u1_xt = np.asarray(u1_xt)
    u2_xt = np.asarray(u2_xt)
    yc_xt = np.asarray(yc_xt)
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, layout= 'constrained', gridspec_kw={'height_ratios': [2.5, 1, 1, 1]})
    ax_field, ax_u1, ax_u2, ax_yc = axes

    #### Flow field

    mesh = ax_field.pcolormesh(x/wake_params.D, y/wake_params.D, frames[0], cmap='plasma',
                        shading='auto', vmin=0, vmax=frames.max())
    fig.colorbar(mesh, ax=ax_field, location = 'top', shrink = 0.5, label = r'$u_1$ [m/s]')
    cl_line, = ax_field.plot(x/wake_params.D, yc_xt[0]/wake_params.D, 'w--', lw=1.2, label = '$y_c$')
    ax_field.set_ylabel(r'$y/D$')
    ax_field.legend(loc='upper right')

    ### flow variable profiles

    u1_line, = ax_u1.plot(x/wake_params.D, u1_xt[0], lw=1.2)
    ax_u1.set_ylim(u1_xt.min(), 1.1*u1_xt.max())
    ax_u1.set_ylabel(r'$u_1$ [m/s]')


    u2_line, = ax_u2.plot(x/wake_params.D, u2_xt[0], lw=1.2)
    ax_u2.set_ylim(1.1*u2_xt.min(), 1.1*u2_xt.max())
    ax_u2.set_ylabel(r'$u_2$ [m/s]')


    yc_line, = ax_yc.plot(x/wake_params.D, yc_xt[0], lw=1.2)
    ax_yc.set_ylim(1.1*yc_xt.min(), 1.1*yc_xt.max())
    ax_yc.set_ylabel(r'$y_c$ [m]')
    ax_yc.set_xlabel(r'$x/D$')

    timestamp = ax_field.text(0.02, 0.92, '', transform = ax_field.transAxes, color='w')

    def update(i):
        mesh.set_array(frames[i].ravel())
        cl_line.set_ydata(yc_xt[i]/wake_params.D)
        u1_line.set_ydata(u1_xt[i])
        u2_line.set_ydata(u2_xt[i])
        yc_line.set_ydata(yc_xt[i])
        timestamp.set_text(rf"$\gamma$ = {wake_params.gamma_deg:.1f}°, t = {ts[i]:.0f} s")
        return mesh, cl_line, u1_line, u2_line, yc_line, timestamp


    ani = FuncAnimation(fig, update, frames=len(ts), interval=50, blit = True)

    save_video(ani,'full_wake_evolution.mp4')