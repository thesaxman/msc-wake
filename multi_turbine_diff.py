""" Trying to visualise the difference seen between using a multi-forcing 
    type of PDE vs a super-imposed solution."""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import jax
import jax.numpy as jnp

from model_params import wake_params
from wake_dynamics import u_point
from video_utils import save_video
from multi_forcing import u1_xt as u1_mf, u2_xt as u2_mf, yc_xt as yc_mf
from super_position import u1_xt as u1_sp, u2_xt as u2_sp, yc_xt as yc_sp

from shapiro_unsteady import delta_u1_0
DELTA_U1_0 = delta_u1_0(wake_params)
u1_xt = u1_mf - u1_sp
u2_xt = u2_mf - u2_sp
yc_xt = yc_mf - yc_sp

from shapiro_unsteady import x as x_grid, ts

y_grid = jnp.linspace(-3*wake_params.D, 3*wake_params.D, 100)

# making animation frames
build_frame = jax.vmap(u_point, in_axes=(0,0,0,None, None), out_axes=1)
all_frames = jax.vmap(build_frame, in_axes=(None,0,0,None, None), out_axes=0)(x_grid, yc_xt, u1_xt, y_grid, wake_params)
assert all_frames.shape == (u1_xt.shape[0], y_grid.size, x_grid.size), all_frames.shape
import numpy as np
frames = np.asarray(all_frames)
u1_xt = np.asarray(u1_xt)/DELTA_U1_0
u2_xt = np.asarray(u2_xt)/DELTA_U1_0
yc_xt = np.asarray(yc_xt)/yc_sp.max()
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, layout= 'constrained', gridspec_kw={'height_ratios': [2.5, 1, 1, 1]})
ax_field, ax_u1, ax_u2, ax_yc = axes

from matplotlib.ticker import MultipleLocator

for ax in axes: # introduce gridlines to make figures 
    ax.xaxis.set_major_locator(MultipleLocator(5.0))
    ax.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax.grid(True, which='major', axis='x', lw=0.6, alpha=0.5)
    ax.grid(True, which='minor', axis='x', lw=0.3, alpha=0.25)

#### Flow field

mesh = ax_field.pcolormesh(x_grid/wake_params.D, y_grid/wake_params.D, frames[0], cmap='plasma',
                    shading='auto', vmin=0, vmax=frames.max())
fig.colorbar(mesh, ax=ax_field, location = 'top', shrink = 0.5, label = r'$u_1/\delta u_{{1,0}}^(0)$')
cl_line, = ax_field.plot(x_grid/wake_params.D, yc_xt[0]/wake_params.D, 'w--', lw=1.2, label = r'$\Delta y_c$')
ax_field.set_ylabel(r'$y/D$')
ax_field.legend(loc='upper right')

### flow variable profiles

u1_line, = ax_u1.plot(x_grid/wake_params.D, u1_xt[0], lw=1.2)
ax_u1.set_ylim(u1_xt.min(), 1.1*u1_xt.max())
ax_u1.set_ylabel(r'$\Delta u_1/\delta u_{{1,0}}^(0)$')


u2_line, = ax_u2.plot(x_grid/wake_params.D, u2_xt[0], lw=1.2)
ax_u2.set_ylim(1.1*u2_xt.min(), 1.1*u2_xt.max())
ax_u2.set_ylabel(r'$\Delta u_2/\delta u_{{1,0}}^(0)$')


yc_line, = ax_yc.plot(x_grid/wake_params.D, yc_xt[0], lw=1.2)
ax_yc.set_ylim(1.1*yc_xt.min(), 1.1*yc_xt.max())
ax_yc.set_ylabel(r'$\Delta y_c/y_{c,max}$ [m]')
ax_yc.set_xlabel(r'$x/D$')

timestamp = ax_field.text(0.02, 0.92, '', transform = ax_field.transAxes, color='w')

from varying_turbine import gamma_t

def update(i):
    mesh.set_array(frames[i].ravel())
    cl_line.set_ydata(yc_xt[i]/wake_params.D)
    u1_line.set_ydata(u1_xt[i])
    u2_line.set_ydata(u2_xt[i])
    yc_line.set_ydata(yc_xt[i])
    timestamp.set_text(rf"$\gamma_1$ = {jnp.degrees(gamma_t(ts[i],wake_params.gamma)):.1f}°, t = {ts[i]:.0f} s")
    return mesh, cl_line, u1_line, u2_line, yc_line, timestamp


ani = FuncAnimation(fig, update, frames=len(ts), interval=50, blit = True)

save_video(ani,'two_turbine_method_diff.mp4')