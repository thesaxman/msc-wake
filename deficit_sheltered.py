""" Sheltered formulation of the forcings, deficit-advection counterpart of sheltered.py:
    turbine 1 is solved with the deficit-aware advection velocity (advects at UINF - u1),
    then turbine 2 sees a locally-reduced UINF equal to turbine 1's upstream deficit at
    turbine 2's location, also solved with the deficit-aware advection velocity."""

__author__ = "Ali Alebeedan"
__date__ = "5/8/2026"

import jax.numpy as jnp
from model_params import wake_params as wp, solver_params as sp, turbines
from unsteady_flow_solver import advecting_sheltered_d_dt, solver
from video_utils import WakeSeries, with_field, full_video


solutions = [(jnp.zeros((sp.nt,sp.nx)), jnp.zeros((sp.nt,sp.nx)), jnp.zeros((sp.nt,sp.nx)))]
y0 = (jnp.zeros((sp.nx,)), jnp.zeros((sp.nx,)), jnp.zeros((sp.nx,)))
for tb in turbines:
    solutions.append(solver(y0, advecting_sheltered_d_dt(tb, sp, solutions),sp).ys)
solutions.pop(0)

if __name__ == "__main__":

    y_grid = jnp.linspace(-3*wp.D, 3*wp.D, 100)

    series_list = with_field(
        [WakeSeries(sp.x_grid, sp.ts, u1_xt, u2_xt, yc_xt, tb)
         for tb, (u1_xt, u2_xt, yc_xt) in zip(turbines, solutions)],
        y_grid,
    )

    full_video(series_list, f'{len(turbines)}_turbine_deficit_sheltered.mp4')
