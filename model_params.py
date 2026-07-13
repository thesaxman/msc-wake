"""
The following script is organises the model parameters used for calculating the
flow field. The parameters are organised in terms of flow field, turbine and 
control parameters.
"""


from wake_dynamics import WakeParams
params = WakeParams(
    D = 126.0, 
    kw = 0.0834, 
    Ct=0.8, 
    gamma_deg =-20.0, 
    UINF=8.0)