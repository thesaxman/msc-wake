""" Trying to visualise the difference seen between using a multi-forcing
    type of PDE vs a super-imposed solution."""

import jax.numpy as jnp

from model_params import solver_params as sp
from unsteady_flow_solver import delta_u1_0, delta_u2_0
from video_utils import WakeSeries, with_field, yaw_label_fn, diff_series, full_video
from multi_forcing import u1_xt as u1_mf, u2_xt as u2_mf, yc_xt as yc_mf
from super_position import u1_xt as u1_sp, u2_xt as u2_sp, yc_xt as yc_sp, turbine1

wp1 = turbine1.wp  # the yawed turbine driving both comparisons

# these are the scaling factors used to normalise the differences between the methods
DELTA_U1_ENVELOPE = float(jnp.max(jnp.abs(delta_u1_0(sp.ts, wp1))))
DELTA_U2_ENVELOPE = float(jnp.max(jnp.abs(delta_u2_0(sp.ts, wp1))))
YC_MAX = float(jnp.max(jnp.abs(jnp.stack([yc_sp, yc_mf]))))

mf_series = WakeSeries(sp.x_grid, sp.ts, u1_mf, u2_mf, yc_mf, wp1)
superpos_series = WakeSeries(sp.x_grid, sp.ts, u1_sp, u2_sp, yc_sp, wp1)

diff = diff_series(
    mf_series, superpos_series,
    u1_scale=DELTA_U1_ENVELOPE, u2_scale=DELTA_U2_ENVELOPE, yc_scale=YC_MAX,
    label_fn=yaw_label_fn([wp1], sp.ts),
)

y_grid = jnp.linspace(-3*wp1.D, 3*wp1.D, 100)
diff = with_field(diff, y_grid)

full_video(diff, 'two_turbine_method_diff.mp4', diff=True, percent=True)
