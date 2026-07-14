"""_summary_ This script is an attempt at adding time-dependence in forcing.
"""
    
__author__ = "Ali Alebeedan"
__date__ = "10/7/2026"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
from model_params import wake_params
from diffrax import diffeqsolve, Tsit5, ODETerm, SaveAt, PIDController


gamma0_rad = jnp.radians(wake_params.gamma_deg)
gamma_amplitude_rad = jnp.radians(30)
gamma_frequency = 0.01827665508523  # Frequency of the sinusoidal variation in Hz

def sinusoidal_variation(t, x0, amplitude, frequency):
    """
    Function to output a sinusoidally time-varying quantity x about mean x = 0.
    
    Parameters:
    t : float
        Time variable/parameter.
    x0 : float
        Initial value.
    amplitude : float
        Amplitude of the variation.
    frequency : float
        Frequency of the variation in Hz.
        
    Returns:
    float
        Time-varying value.
    """
    offset = jnp.arcsin(x0 / amplitude) / (2 * jnp.pi * frequency)
    return amplitude * jnp.sin(2 * jnp.pi * frequency * (t + offset))

def gamma_t(t, gamma0):
    """Time varying function of the yaw angle.

    Args:
        t (Float): time variable
        gamma0 (Float): initial yaw angle
    """
    
    return sinusoidal_variation(t, gamma0, amplitude=gamma_amplitude_rad, frequency=gamma_frequency)

from shapiro_unsteady import solver

def rhs(t, state, args):
    
    u1, u2, yc = state
    
    