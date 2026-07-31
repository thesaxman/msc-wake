"""
The following script is organises the model parameters used for calculating the
flow field. The parameters are collected as wake dynamics parameters and solver scheme parameters.
"""
from functools import partial

from wake_dynamics import WakeParams, make_turbine, sinusoid_gamma_t
from unsteady_flow_solver import SolverParams

wake_params = WakeParams(
    D = 0.15, 
    kw = 0.0834, 
    Ct=0.8, 
    gamma_deg = 0, 
    UINF=4.88)
solver_params = SolverParams(wp= wake_params)

mk = partial(make_turbine, wake_params)

turbine1 = mk(0.0, gamma_deg=-15.0)
turbine2 = mk(5.0, gamma_fn=sinusoid_gamma_t)
#turbine3 = mk(10.0, gamm)

turbines = [turbine1, turbine2]