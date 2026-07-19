""" Trying to visualise the difference seen between using a multi-forcing 
    type of PDE vs a super-imposed solution."""

import dataclasses
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import ticker

import jax
import jax.numpy as jnp

from model_params import wake_params
from wake_dynamics import u_point
from video_utils import save_video
from multi_forcing import u1_xt as u1_mf, u2_xt as u2_mf, yc_xt as yc_mf
from super_position import u1_xt as u1_sp, u2_xt as u2_sp, yc_xt as yc_sp

from wake_dynamics import gamma_t

wake_params = dataclasses.replace(wake_params, gamma_fn = gamma_t)

u1_xt = u1_mf - u1_sp
u2_xt = u2_mf - u2_sp
yc_xt = yc_mf - yc_sp

from shapiro_unsteady import x as x_grid, ts
from unsteady_flow_solver import delta_u1_0, delta_u2_0
# these are the scaling factors used to normalise the differences between the methods
DELTA_U1_ENVELOPE = float(jnp.max(jnp.abs(delta_u1_0(ts,wake_params))))
DELTA_U2_ENVELOPE = float(jnp.max(jnp.abs(delta_u2_0(ts,wake_params))))
YC_MAX = np.asarray(jnp.max(jnp.abs(jnp.stack([yc_sp, yc_mf]))))

y_grid = jnp.linspace(-3*wake_params.D, 3*wake_params.D, 100)

# making animation frames
build_frame = jax.vmap(u_point, in_axes=(0,0,0,None, None), out_axes=1)
all_frames = jax.vmap(build_frame, in_axes=(None,0,0,None, None), out_axes=0)(x_grid, yc_xt, u1_xt, y_grid, wake_params)
assert all_frames.shape == (u1_xt.shape[0], y_grid.size, x_grid.size), all_frames.shape
frames = np.asarray(all_frames)/DELTA_U1_ENVELOPE
u1_xt_norm = np.asarray(u1_xt)/DELTA_U1_ENVELOPE
u2_xt_norm = np.asarray(u2_xt)/DELTA_U2_ENVELOPE
yc_xt_norm = np.asarray(yc_xt)/YC_MAX
 
                

fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, layout= 'constrained', gridspec_kw={'height_ratios': [2.5, 1, 1, 1]})
ax_field, ax_u1, ax_u2, ax_yc = axes

from matplotlib.ticker import MultipleLocator

for ax in axes: # introduce gridlines to make figures readable
    ax.xaxis.set_major_locator(MultipleLocator(5.0))
    ax.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax.grid(True, which='major', axis='x', lw=0.6, alpha=0.5)
    ax.grid(True, which='minor', axis='x', lw=0.3, alpha=0.25)
    
for ax in axes[1:]: # setting y-axes of variable profiles to percentage
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

#### Flow field

m = np.abs(frames).max()
mesh = ax_field.pcolormesh(x_grid/wake_params.D, y_grid/wake_params.D, frames[0], cmap='RdBu_r', shading='auto', vmin=-m, vmax=m)
fig.colorbar(mesh, ax=ax_field, location = 'top', shrink = 0.5, label = r'$\Delta u_1/\delta u_{1,env}$', format = ticker.PercentFormatter(xmax=1.0))
cl_line, = ax_field.plot(x_grid/wake_params.D, yc_xt[0]/wake_params.D, 'w--', lw=1.2, label = r'$\Delta y_c$')
ax_field.set_ylabel(r'$y/D$')
ax_field.legend(loc='upper right')

### flow variable profiles

u1_line, = ax_u1.plot(x_grid/wake_params.D, u1_xt_norm[0], lw=1.2)
ax_u1.set_ylim(u1_xt_norm.min(), 1.1*u1_xt_norm.max())
ax_u1.set_ylabel(r'$\Delta u_1/\delta u_{1,\mathrm{env}}$')


u2_line, = ax_u2.plot(x_grid/wake_params.D, u2_xt_norm[0], lw=1.2)
ax_u2.set_ylim(1.1*u2_xt_norm.min(), 1.1*u2_xt_norm.max())
ax_u2.set_ylabel(r'$\Delta u_2/\delta u_{2,\mathrm{env}}$')


yc_line, = ax_yc.plot(x_grid/wake_params.D, yc_xt_norm[0], lw=1.2)
ax_yc.set_ylim(1.1*yc_xt_norm.min(), 1.1*yc_xt_norm.max())
ax_yc.set_ylabel(r'$\Delta y_c/y_{c,\mathrm{max}}$')
ax_yc.set_xlabel(r'$x/D$')

timestamp = ax_field.text(0.02, 0.92, '', transform = ax_field.transAxes, color='w')

def update(i):
    mesh.set_array(frames[i].ravel())
    cl_line.set_ydata(yc_xt[i]/wake_params.D)
    u1_line.set_ydata(u1_xt_norm[i])
    u2_line.set_ydata(u2_xt_norm[i])
    yc_line.set_ydata(yc_xt_norm[i])
    timestamp.set_text(rf"$\gamma_1$ = {jnp.degrees(gamma_t(ts[i],wake_params.gamma)):.1f}°, t = {ts[i]:.0f} s")
    return mesh, cl_line, u1_line, u2_line, yc_line, timestamp


ani = FuncAnimation(fig, update, frames=len(ts), interval=50, blit = True)

save_video(ani,'two_turbine_method_diff.mp4')