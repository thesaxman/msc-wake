import jax
import jax.numpy as jnp
import equinox as eqx


def wake_expansion(x, D, kw): # General wake expansion function (might later add an implementation for varying kw)
    return 1+kw*jnp.log(1+jnp.exp(2*(x-D)/D))

def wake_area(x, D, kw): # Wake area function:
    return jnp.pi * D ** 2 / 4 * wake_expansion(x, D, kw) ** 2

def d_wake_area_dx(x, D, kw): # x derivative of the wake area expansion
    return jax.grad(wake_area, argnums = 0)(x, D, kw)

def gaussian_forcing(x, D): # Gaussian forcing function replacing Dirac delta function
    return 1/(jnp.sqrt(2*jnp.pi)*D/2) * jnp.exp(-0.5 * (x)**2 / (D/2)**2)

class WakeParams(eqx.Module):
    D:      float
    kw:     float
    Ct:     float
    gamma_deg: float
    UINF:   float
    sigma0_ratio: float = 0.235
    boundary_diams: float = 20.0
    
    @property
    def sigma0(self):
        return self.sigma0_ratio* self.D
    
    @property
    def boundary(self):
        return self.boundary_diams * self.D
    
    @property
    def upstream_bound(self):
        return self.D *-4.0

    
def dw(x, p:WakeParams):
    return wake_expansion(x, p.D, p.kw)

def A(x, p: WakeParams):
    return wake_area(x, p.D, p.kw)

def dA_dx(x, p: WakeParams):
    return d_wake_area_dx(x, p.D, p.kw)

def G(x, p:WakeParams):
    return gaussian_forcing(x,p.D)