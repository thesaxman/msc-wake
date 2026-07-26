"""
The following script is organises the model parameters used for calculating the
flow field. The parameters are collected as wake dynamics parameters and solver scheme parameters.
"""


from wake_dynamics import WakeParams
from unsteady_flow_solver import SolverParams
wake_params = WakeParams(
    D = 0.15, 
    kw = 0.0834, 
    Ct=0.8, 
    gamma_deg = 0, 
    UINF=4.88)
solver_params = SolverParams(wp= wake_params)