import jax.random as jr
from diffrax import diffeqsolve, ControlTerm, Euler, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree
import jax.numpy as jnp
import matplotlib.pyplot as plt
from datetime import date
R = 130

key = jr.PRNGKey(date.today().day)  # Use the current day as the seed for reproducibility

def generate_disturbances(R, ymax, key):
    drift = lambda t, y, args: -y
    diffusion = lambda t, y, args: 0.1 * t
    brownian_motion = VirtualBrownianTree(0, R, tol=1e-3, shape=(), key=key)
    terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, brownian_motion))
    solver = Euler()
    saveat = SaveAt(dense=True)
    sol = diffeqsolve(terms, solver, 0, R, dt0=0.05, y0=0., saveat=saveat)
    return jnp.clip(sol.evaluate(jnp.linspace(0, R, 100)), -ymax, ymax)
    #return sol.evaluate(jnp.linspace(0, R, 100))

key, subkey = jr.split(key)
u1 = generate_disturbances(R, 6., subkey)
key, subkey = jr.split(key)
u2 = generate_disturbances(R, 6., subkey)
plt.plot(jnp.linspace(0, R, 100), u1, label='Disturbance 1')
plt.plot(jnp.linspace(0, R, 100), u2, label='Disturbance 2')
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Generated Disturbances (Clipped)")
plt.legend()
plt.show()