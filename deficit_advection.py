""" This approach to implementing an advection velocity that takes into account the deficit rather than purely free-stream"""

__author__ = "Ali Alebeedan"
__date__ = "25/7/2026"

from functools import partial
import dataclasses

import jax.numpy as jnp
from model_params import wake_params as wp, solver_params as sp
from wake_dynamics import make_turbine
from unsteady_flow_solver import advecting_d_dt, solver, adv_S1, adv_S2

from video_utils import WakeSeries, with_field, yaw_label_fn, full_video


mk = partial(make_turbine, wp)

turbines = [
    mk(0.0, gamma_deg = -15.0)
]

sp = dataclasses.replace(sp, max_steps  = 10_000_000, S1 = adv_S1, S2 = adv_S2)

y0 = (jnp.zeros(sp.nx), jnp.zeros(sp.nx), jnp.zeros(sp.nx))

u1_xt, u2_xt, yc_xt = solver(y0=y0, rhs_func=advecting_d_dt(turbines, sp), sp=sp).ys

print(float((wp.UINF - u1_xt).min()))

if __name__ == "__main__":
    D = wp.D
    x_grid = sp.x_grid
    y_grid = jnp.linspace(-3.0*D, 3*D, 100)

    series = with_field(
        WakeSeries(x_grid, sp.ts, u1_xt, u2_xt, yc_xt, wp,
                   label_fn=yaw_label_fn([tb.wp for tb in turbines], sp.ts)),
        y_grid,
    )
    g = turbines[0].wp.gamma_deg
    sign = "neg" if g < 0 else "pos"
    g = f"{g:.1f}".replace('.', 'p')
    full_video(series, f'advecting_deficit_gamma_{sign}{g}.mp4')
