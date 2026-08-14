""" Trying to visualise the difference seen between using a multi-forcing
    type of PDE vs a super-imposed solution."""

import jax.numpy as jnp

from model_params import solver_params as sp, turbines
from unsteady_flow_solver import du1_0, du2_0
from video_utils import WakeSeries, with_field, yaw_label_fn, diff_series, full_video
from multi_forcing import u1_xt as u1_mf, u2_xt as u2_mf, yc_xt as yc_mf
from super_position import u1_xt as u1_sp, u2_xt as u2_sp, yc_xt as yc_sp
#from sheltered import u1_xt as u1_sh, u2_xt as u2_sh, yc_xt as yc_sh
from deficit_advection_mf import u1_xt as u1_advmf, u2_xt as u2_advmf, yc_xt as yc_advmf
from deficit_advection_sp import u1_xt as u1_advsp, u2_xt as u2_advsp, yc_xt as yc_advsp

tb1 = turbines[1]  # the yawed turbine driving both comparisons
wp1 = tb1.wp

# these are the scaling factors used to normalise the differences between the methods
DELTA_U1_ENVELOPE = float(jnp.max(jnp.abs(du1_0(sp.ts, wp1))))
DELTA_U2_ENVELOPE = float(jnp.max(jnp.abs(du2_0(sp.ts, wp1))))
YC_MAX = float(jnp.max(jnp.abs(jnp.stack([yc_mf, yc_sp]))))

shapiro_mf_series = WakeSeries(sp.x_grid, sp.ts, u1_mf, u2_mf, yc_mf, tb1)
shapiro_sp_series = WakeSeries(sp.x_grid, sp.ts, u1_sp, u2_sp, yc_sp, tb1)
advected_mf_series = WakeSeries(sp.x_grid, sp.ts, u1_advmf, u2_advmf, yc_advmf, tb1)
advected_sp_series = WakeSeries(sp.x_grid, sp.ts, u1_advsp, u2_advsp, yc_advsp, tb1)
#sheltered_series = WakeSeries(sp.x_grid, sp.ts, u1_sh, u2_sh, yc_sh, tb1)


diff = diff_series(
    shapiro_mf_series, shapiro_sp_series,
    u1_scale=DELTA_U1_ENVELOPE, u2_scale=DELTA_U2_ENVELOPE, yc_scale=YC_MAX,
)

y_grid = jnp.linspace(-3*wp1.D, 3*wp1.D, 100)
diff = with_field(diff, y_grid)

full_video(diff, 'two_turbine_shapiro_mf_v_sp.mp4', diff=True, percent=True,
           label_fn=yaw_label_fn([tb.wp for tb in turbines], sp.ts))
