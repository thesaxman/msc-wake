"""
As a variant to stead_state.py, this file imports the same model parameters, but uses a different approach to solve the ODEs for the wake centreline deflection and velocity deficit. The results are plotted in a similar manner to steady_state.py.
"""

__author__ = "Ali Alebeedan"
__date__ = "2/7/2026"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, Tsit5, ODETerm, SaveAt, PIDController

from model_params import params
from wake_dynamics import dw, A, dA_dx, G

""" def wake_expansion(x, D, kw): # General wake expansion function
    return 1+kw*jnp.log(1+jnp.exp(2*(x-D)/D))
def dw(x): # Wake expansion with flow parameters specified
    return wake_expansion(x, D, kw)
def A(x): # Wake area function
    return jnp.pi * D ** 2 / 4 * dw(x) ** 2
dA_dx = jax.grad(A)
gamma_rad = jnp.radians(gamma)
cos_gamma = jnp.cos(gamma_rad)
sin_gamma = jnp.sin(gamma_rad)
def G(x):
    return 1/(jnp.sqrt(2*jnp.pi)*D/2) * jnp.exp(-0.5 * (x)**2 / (D/2)**2)
 """

gamma_rad = jnp.radians(params.gamma_deg)
cos_gamma = jnp.cos(gamma_rad)
sin_gamma = jnp.sin(gamma_rad)

def solve_steady_u1(p = params):

    def du1_dx(x, u1, args): # ODE for u1 definition
        return -dA_dx(x, params)/A(x, params)*u1 + params.UINF*(1-jnp.sqrt(1-Ct*cos_gamma**2))*G(x, params)

    u1_sol = diffeqsolve(ODETerm(du1_dx), Tsit5(), t0=-4.0*params.D, t1=float(p.boundary()), dt0=0.01, y0=0.0, saveat=SaveAt(dense=True), stepsize_controller=PIDController(rtol=1e-5, atol=1e-5))

    return jax.vmap(u1_sol.evaluate)

def solve_steady_u2(p = params):

    def du2_dx(x, u1, args): # ODE for u2 definition
        return -dA_dx(x, p)/A(x, p)*u1 + \
            p.UINF*(1.0/4.0*p.Ct*cos_gamma**2*sin_gamma) \
            *G(x,p)

    u2_sol = diffeqsolve(ODETerm(du2_dx), Tsit5(), t0=-4.0*params.D, t1=float(p.boundary()), dt0=0.01, y0=0.0, saveat=SaveAt(dense=True), stepsize_controller=PIDController(rtol=1e-5, atol=1e-5))

    return jax.vmap(u2_sol.evaluate)

def solve_steady_yc(p = params):

    u2 = solve_steady_u2
    def dyc_dx(x, y, args):
    # This function defines the ODE for the wake centreline deflection
        
        return -u2(x*cos_gamma) / p.UINF

    yc_sol = diffeqsolve(ODETerm(dyc_dx), Tsit5(), t0=-2.0*p.D, t1=float(p.boundary()), dt0=0.01, y0=0., saveat=SaveAt(dense=True), stepsize_controller=PIDController(rtol=1e-5, atol=1e-5))

    return jax.vmap(yc_sol.evaluate)


if __name__ == "__main__":

    #Flow field can be first evaluated along x
    x = jnp.linspace(0.0, float(boundary), 100)
    y = jnp.linspace(-1.1*D, 1.1*D, 100)
    X, Y = jnp.meshgrid(x, y)
    yc_x1 = yc_steady1(x)
    yc_x2 = yc_steady2(x)
    u1_x = u1_steady(x)
    u2_x = u2_steady(x)
    # Wake expansion effects onto 2D
    def sigma_y(x):
        return sigma0 * dw(x)
    def gaussian(x, yc, y):
        return 0.5 * (D / 2 / sigma0)**2 * \
            jnp.exp(-0.5 * ((y - yc) / sigma_y(x))**2)
    #gaussian_2d = jax.vmap(gaussian, in_axes=(0,0, None), out_axes=1)(x, yc_x, y)
    def u1_point(x, yc, u1, y):
        return u1 * gaussian(x, yc, y)

    u1_field = jax.vmap(u1_point, in_axes=(0,0,0,None), out_axes=1)(x, yc_x1, u1_x, y)

    # --- plot ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # velocity deficit field
    cf = axes[0].pcolormesh(X/D, Y/D, u1_field, cmap='RdBu_r', shading='auto')
    axes[0].plot(x/D, yc_x1/D, color='white', linestyle='dashed', linewidth=1.5,
                label='Wake centreline $y_c(x)$')
    axes[0].plot(x/D, yc_x2/D, color='yellow', linestyle='dashed', linewidth=1.5,
                label='Wake centreline $y_c(x)$ (2nd ODE)')
    plt.colorbar(cf, ax=axes[0], label='$u(x,y)$ [m/s]')
    axes[0].set_ylabel('$y/D$')
    axes[0].set_title('Streamwise velocity deficit ($\gamma$ = {:.1f}°)'.format(gamma))
    #axes[0].legend()

    # centreline trajectory alone
    axes[2].plot(x/D, yc_x1/D, color='steelblue', linestyle='dashed', linewidth=1.5,
                label='Wake centreline $y_c(x)$')
    axes[2].plot(x/D, yc_x2/D, color='yellow', linestyle='dashed', linewidth=1.5,
                label='Wake centreline $y_c(x)$ (2nd ODE)')
    axes[2].set_xlabel('$x/D$')
    axes[2].set_ylabel('$y_c/D$')
    axes[2].set_title('Wake centreline deflection')
    axes[2].legend()

    # closed form solution for comparison
    u1_0 = UINF*(1-jnp.sqrt(1-Ct*cos_gamma**2))

    u1_closed = u1_0/(dw(x)**2)*0.5*(1+ jax.scipy.special.erf(x/(D/2*jnp.sqrt(2))))

    u1_closed_field = jax.vmap(u1_point, in_axes=(0,0,0,None), out_axes=1)(x, yc_x1, u1_closed, y)
    cf = axes[1].pcolormesh(X/D, Y/D, u1_field - u1_closed_field, cmap='RdBu_r', shading='auto')
    plt.colorbar(cf, ax=axes[1], label='$u(x,y)$ [m/s]')
    axes[1].set_ylabel('$y/D$')
    axes[1].set_title('Difference in velocity deficit to closed form solution')



    plt.tight_layout()
    plt.savefig("wake_field_2d.png", dpi=150)
    ##plt.show()