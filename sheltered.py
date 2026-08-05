""" An idea of introducing a sheltered formulation of the forcings"""

__author__ = "Ali Alebeedan"
__date__ = "29/7/2026"

import dataclasses

import jax.numpy as jnp
from model_params import wake_params as wp, solver_params as sp, turbines
from wake_dynamics import make_turbine, sinusoid_gamma_t
from unsteady_flow_solver import default_d_dt, solver, S1_default, S2_default

from video_utils import WakeSeries, with_field, yaw_label_fn, full_video


y0 = (jnp.zeros(sp.nx), jnp.zeros(sp.nx), jnp.zeros(sp.nx))

u1_xt1, u2_xt1, yc_xt1 = solver(y0, default_d_dt([turbines[0]], sp), sp).ys

i2 = int(jnp.argmin(jnp.abs(sp.x_grid-turbines[1].x0)))

def S1_sheltered(t, u1, p):
    #upstream deficit at turbine 2's location at time t
    u1_up = jnp.interp(t, sp.ts, u1_xt1[:, i2])
    p_local = dataclasses.replace(p, UINF=p.UINF - u1_up)
    return S1_default(t, u1, p_local)

def S2_sheltered(t, u1, p):
    #upstream deficit at turbine 2's location at time t
    u1_up = jnp.interp(t, sp.ts, u1_xt1[:, i2])
    p_local = dataclasses.replace(p, UINF=p.UINF - u1_up)
    return S2_default(t, u1, p_local)

sp2 = dataclasses.replace(sp, S1=S1_sheltered, S2=S2_sheltered)

u1_xt2, u2_xt2, yc_xt2 = solver(y0, default_d_dt([turbines[1]], sp2), sp2).ys
u1_xt = u1_xt1 + u1_xt2
u2_xt = u2_xt1 + u2_xt2
yc_xt = yc_xt1 + yc_xt2

if __name__ == "__main__":

    y_grid = jnp.linspace(-3*wp.D, 3*wp.D, 100)

    series = with_field(
        WakeSeries(sp.x_grid, sp.ts, u1_xt, u2_xt, yc_xt, wp,
                   label_fn=yaw_label_fn([tb.wp for tb in turbines], sp.ts)),
        y_grid,
    )

    full_video(series, 'two_turbine_sheltered.mp4')
