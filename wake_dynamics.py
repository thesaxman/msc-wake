from collections.abc import Callable
import dataclasses

import jax
import jax.numpy as jnp
import equinox as eqx


def wake_expansion(x, D, kw): # General wake expansion function (might later add an implementation for varying kw)
    z = 2.0 * (x-D)/D
    return 1+kw*jnp.logaddexp(0.0,z)

def wake_area(x, D, kw): # Wake area function:
    return jnp.pi * D ** 2 / 4 * wake_expansion(x, D, kw) ** 2

_d_wake_area_dx = jax.grad(wake_area, argnums=0) # less expensive to calculate here
def d_wake_area_dx(x, D, kw): # x derivative of the wake area expansion
    return _d_wake_area_dx(x, D, kw)

def gaussian_forcing(x, D): # Gaussian forcing function replacing Dirac delta function
    return 1/(jnp.sqrt(2*jnp.pi)*D/2) * jnp.exp(-0.5 * (x)**2 / (D/2)**2)

def constant_gamma(t, gamma):
    return gamma

class WakeParams(eqx.Module):
    D:      float
    kw:     float
    Ct:     float
    gamma_deg: float
    UINF:   float
    sigma0_ratio: float = 0.235
    boundary_diams: float = 20.0
    upstream_diams: float = -4.0
    gamma_fn: Callable = eqx.field(static = True, default = constant_gamma)
    max_gamma_deg: float = 30.0
    
    
    @property
    def sigma0(self):
        return self.sigma0_ratio* self.D
    
    @property
    def boundary(self):
        return self.boundary_diams * self.D
    
    @property
    def upstream_bound(self):
        return self. upstream_diams * self.D
    
    @property
    def t1(self):
        return 2.0 * (self.boundary - self.upstream_bound)/self.UINF
    @property
    def gamma(self):
        return jnp.radians(self.gamma_deg)
    
    def gamma_at(self, t):
        """Yaw angle (in radians) at time t, via the configured control law."""
        return self.gamma_fn(t, self.gamma)

class Turbine(eqx.Module):
    wp: WakeParams
    x0: float = eqx.field(static = True, default = 0.0)

def make_turbine(base: WakeParams, x0_diams: float, *,
                 gamma_deg=None, gamma_fn=None):
    """Build a Turbine from a base WakeParams, overriding only what's given.

    Args:
        base:      the shared baseline WakeParams
        x0_diams:  streamwise position in rotor diameters
        gamma_deg: override base yaw angle (degrees), if given
        gamma_fn:  override control law, if given
    """
    overrides = {}
    if gamma_deg is not None:
        overrides['gamma_deg'] = gamma_deg
    if gamma_fn is not None:
        overrides['gamma_fn'] = gamma_fn
    wp = dataclasses.replace(base, **overrides) if overrides else base
    return Turbine(wp=wp, x0=x0_diams * base.D)

def dw(x, p: WakeParams):
    return wake_expansion(x, p.D, p.kw)

def A(x, p: WakeParams):
    return wake_area(x, p.D, p.kw)

def dA_dx(x, p: WakeParams):
    return d_wake_area_dx(x, p.D, p.kw)

def expansion(x_grid, p): # to compute the expansion term w(x)/UINF (normalised wrt UINF so may need further fleshing out for future local velocity expansion)
    return jax.vmap(dA_dx, in_axes=(0,None))(x_grid,p) / A(x_grid,p) 

def G(x, p: WakeParams):
    return gaussian_forcing(x,p.D)

def sigma_y(x, p: WakeParams):
    return p.sigma0 * dw(x,p)

def gaussian(x, yc, y, p: WakeParams):
    return 0.5 * (p.D / 2 / p.sigma0)**2 * \
        jnp.exp(-0.5 * ((y - yc) / sigma_y(x, p))**2)

def u_point(x, yc, u1, y, p: WakeParams):
    return u1 * gaussian(x, yc, y, p)

MAX_YAW = float(jnp.radians(30)) # Max operating yaw taken as 30 degrees
YAW_FREQUENCY = 8.333e-4  # NREL nominal yaw-rate normalised by full rotation for frequency

def sinusoidal_variation_t(t, x0, amplitude, frequency):
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
        Time-varying value resolved at t.
    """
    offset = jnp.arcsin(x0 / amplitude) / (2 * jnp.pi * frequency)
    return amplitude * jnp.sin(2 * jnp.pi * frequency * (t + offset))

def sinusoid_gamma_t(t, gamma0, amp = MAX_YAW, freq = YAW_FREQUENCY):
    """Time varying sinusoidal function of the yaw angle.

    Args:
        t (Float): time variable
        gamma0 (Float): initial yaw angle (rad)
        gamma_amp (float): yaw variation amplitude (rad)
        gamma_freq (float): yaw variation frequency (Hz)
    """
    
    return sinusoidal_variation_t(t, gamma0, amplitude=amp, frequency=freq)

