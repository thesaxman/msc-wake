from collections.abc import Callable

import jax
import jax.numpy as jnp
import equinox as eqx


def wake_expansion(x, D, kw): # General wake expansion function (might later add an implementation for varying kw)
    z = 2.0 * (x-D)/D
    return 1+kw*jnp.logaddexp(0.0,z)

def wake_area(x, D, kw): # Wake area function:
    return jnp.pi * D ** 2 / 4 * wake_expansion(x, D, kw) ** 2

def d_wake_area_dx(x, D, kw): # x derivative of the wake area expansion
    return jax.grad(wake_area, argnums = 0)(x, D, kw)

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
    gamma_fn: Callable = eqx.field(static = True, default = constant_gamma)
    
    @property
    def sigma0(self):
        return self.sigma0_ratio* self.D
    
    @property
    def boundary(self):
        return self.boundary_diams * self.D
    
    @property
    def upstream_bound(self):
        return self.D *-4.0
    
    @property
    def t1(self):
        return 2.0 * (self.boundary - self.upstream_bound)/self.UINF
    @property
    def gamma(self):
        return jnp.radians(self.gamma_deg)
    

    
def dw(x, p:WakeParams):
    return wake_expansion(x, p.D, p.kw)

def A(x, p: WakeParams):
    return wake_area(x, p.D, p.kw)

def dA_dx(x, p: WakeParams):
    return d_wake_area_dx(x, p.D, p.kw)

def G(x, p:WakeParams):
    return gaussian_forcing(x,p.D)

def sigma_y(x, p):
    return p.sigma0 * dw(x,p)

def gaussian(x, yc, y, p):
    return 0.5 * (p.D / 2 / p.sigma0)**2 * \
        jnp.exp(-0.5 * ((y - yc) / sigma_y(x, p))**2)

def u_point(x, yc, u1, y, p):
    return u1 * gaussian(x, yc, y, p)