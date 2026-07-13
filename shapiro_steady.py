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


def solve_steady_u1(p):
    
    gamma_rad = jnp.radians(p.gamma_deg)
    cos_gamma = jnp.cos(gamma_rad)
    S1 = p.UINF*(1-jnp.sqrt(1-p.Ct*cos_gamma**2))
    def du1_dx(x, u1, args): # ODE for u1 definition
        return -dA_dx(x, p)/A(x, p)*u1 + S1*G(x, p)

    u1_sol = diffeqsolve(ODETerm(du1_dx), Tsit5(), t0=p.upstream_bound, t1=p.boundary, dt0=0.01, y0=0.0, saveat=SaveAt(dense=True), stepsize_controller=PIDController(rtol=1e-5, atol=1e-7))

    return u1_sol.evaluate

def solve_steady_u2(p):
    gamma_rad = jnp.radians(p.gamma_deg)
    cos_gamma = jnp.cos(gamma_rad)
    sin_gamma = jnp.sin(gamma_rad)

    def du2_dx(x, u2, args): # ODE for u2 definition
        return -dA_dx(x, p)/A(x, p)*u2 + \
            p.UINF*(1.0/4.0*p.Ct*cos_gamma**2*sin_gamma) \
            *G(x,p)

    u2_sol = diffeqsolve(ODETerm(du2_dx), Tsit5(), t0=p.upstream_bound, t1=p.boundary, dt0=0.01, y0=0.0, saveat=SaveAt(dense=True), stepsize_controller=PIDController(rtol=1e-5, atol=1e-5))

    return u2_sol.evaluate

def solve_steady_yc(p):
    
    u2 = solve_steady_u2(p=p)
    # This function defines the ODE for the wake centreline deflection
    def dyc_dx(x, y, args):
        return -u2(x) / p.UINF # taking streamwise aligned coordinate
    
    yc_sol = diffeqsolve(ODETerm(dyc_dx), Tsit5(), t0=p.upstream_bound, t1=p.boundary, dt0=0.01, y0=0., saveat=SaveAt(dense=True), stepsize_controller=PIDController(rtol=1e-5, atol=1e-5))

    return yc_sol.evaluate



if __name__ == "__main__":

    #Flow field can be first evaluated along x
    x = jnp.linspace(0.0, params.boundary, 100)
    y = jnp.linspace(-1.1*params.D, 1.1*params.D, 100)
    yc_x = jax.vmap(solve_steady_yc(params))(x)/params.D
    u1_x = jax.vmap(solve_steady_u1(params))(x)
    u2_x = jax.vmap(solve_steady_u2(params))(x)
    X, Y = jnp.meshgrid(x/params.D, y/params.D) # gridspace normalised with rotor diam
    # Wake expansion effects onto 2D
    from wake_dynamics import dw
    def sigma_y(x, p):
        return p.sigma0 * dw(x,p)
    def gaussian(x, yc, y, p):
        return 0.5 * (p.D / 2 / p.sigma0)**2 * \
            jnp.exp(-0.5 * ((y - yc) / sigma_y(x, p))**2)
    def u1_point(x, yc, u1, y, p):
        return u1 * gaussian(x, yc, y, p)

    u1_field = jax.vmap(u1_point, in_axes=(0,0,0,None,None), out_axes=1)(x, yc_x, u1_x, y,params)

    # --- plot ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    x,y = x/params.D,y/params.D
    yc_x = yc_x/params.D
    # velocity deficit field
    cf = axes[0].pcolormesh(X, Y, u1_field/params.UINF, cmap='RdBu_r', shading='auto')
    axes[0].plot(x, yc_x, color='white', linestyle='dashed', linewidth=1.5,
                label='Wake centreline $y_c(x)$')
    plt.colorbar(cf, ax=axes[0], label='$u(x,y)/U_\infty$')
    axes[0].set_ylabel('$y/D$')
    axes[0].set_title('Streamwise velocity deficit ($\gamma$ = {:.1f}°)'.format(params.gamma_deg))
    axes[0].legend()

    # centreline trajectory alone
    axes[2].plot(x, yc_x, color='steelblue', linestyle='dashed', linewidth=1.5,
                label='Wake centreline $y_c(x)$')
    axes[2].set_xlabel('$x/D$')
    axes[2].set_ylabel('$y_c/D$')
    axes[2].set_title('Wake centreline deflection')
    axes[2].legend()

    # closed form solution for comparison
    
    gamma_rad = jnp.radians(params.gamma_deg)
    cos_gamma = jnp.cos(gamma_rad)
    sin_gamma = jnp.sin(gamma_rad)
    
    u1_0 = params.UINF*(1-jnp.sqrt(1-params.Ct*cos_gamma)**2)

    u1_closed = u1_0/(dw(x,params)**2)*0.5*(1+ jax.scipy.special.erf(x/(params.D/2*jnp.sqrt(2))))
    
    u2_0 = params.UINF*(0.25 * params.Ct * cos_gamma**2 * sin_gamma)

    u2_closed = u2_0/(dw(x,params)**2)*0.5*(1+ jax.scipy.special.erf(x/(params.D/2*jnp.sqrt(2))))
    def dyc_dx(x, y, args):
        return -u2(x) / p.UINF # taking streamwise aligned coordinate
    
    yc_sol = diffeqsolve(ODETerm(dyc_dx), Tsit5(), t0=p.upstream_bound, t1=p.boundary, dt0=0.01, y0=0., saveat=SaveAt(dense=True), stepsize_controller=PIDController(rtol=1e-5, atol=1e-5))
    yc_closed = jax.vmap(yc_sol.evaluate)(x)
    u1_closed_field = jax.vmap(u1_point, in_axes=(0,0,0,None,None), out_axes=1)(x, yc_closed, u1_closed, y, params)
    cf = axes[1].pcolormesh(X, Y, (u1_field - u1_closed_field)/params.UINF, cmap='RdBu_r', shading='auto')
    plt.colorbar(cf, ax=axes[1], label='$u(x,y)/U_\infty$')
    axes[1].set_ylabel('$y/D$')
    axes[1].set_title('Difference in velocity deficit to closed form solution')



    plt.tight_layout()
    #plt.savefig("outputs/wake_field_2d.png", dpi=150)
    plt.show()