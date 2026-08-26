"""
    Developing a method to calculate effective yaw angle for a wind turbine based on wind direction and turbine orientation.
"""

__author__ = "Ali Alebeedan"
__date__ = "07/08/2026"

import jax.numpy as jnp
from model_params import wake_params as wp, solver_params as sp, turbines
from unsteady_flow_solver import solver, sheltered_d_dt, advecting_sheltered_d_dt

from video_utils import WakeSeries, with_field, full_video

def normal():
    solutions = [(jnp.zeros((sp.nt,sp.nx)), jnp.zeros((sp.nt,sp.nx)), jnp.zeros((sp.nt,sp.nx)))]
    y0 = (jnp.zeros((sp.nx,)), jnp.zeros((sp.nx,)), jnp.zeros((sp.nx,)))
    for tb in turbines:
        solutions.append(solver(y0, sheltered_d_dt(tb, sp, solutions, skew=True),sp).ys)
    return solutions.pop(0)

def advecting():
    solutions = [(jnp.zeros((sp.nt,sp.nx)), jnp.zeros((sp.nt,sp.nx)), jnp.zeros((sp.nt,sp.nx)))]
    y0 = (jnp.zeros((sp.nx,)), jnp.zeros((sp.nx,)), jnp.zeros((sp.nx,)))
    for tb in turbines:
        solutions.append(solver(y0, advecting_sheltered_d_dt(tb, sp, solutions, skew=True),sp).ys)
    return solutions.pop(0)

if __name__ == "__main__":

    #solutions = normal()
    solutions = advecting()
    y_grid = jnp.linspace(-3*wp.D, 3*wp.D, 100)

    series_list = with_field(
        [WakeSeries(sp.x_grid, sp.ts, u1_xt, u2_xt, yc_xt, tb)
         for tb, (u1_xt, u2_xt, yc_xt) in zip(turbines, solutions)],
        y_grid,
    )

    full_video(series_list, f'{len(turbines)}_tb_adv_shel_skew.mp4')
