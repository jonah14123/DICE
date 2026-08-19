"""
This script imports the physics and numerical approximation for integration from physics_intgration.py, and creates a dataset

make_dataset(physics parameters, initial position, random_key, integration params, resolution) -> pupulation samples at resolution over T

*NOTE we vary the rotational component by moving the well, NOT C, hence x0 is generated about C

makedataset()
"""
from physics_integration import CENTER, center, drift, dq, euler_maruyama #to generate x0
import jax.numpy as jnp
import jax

def make_init(key: jax.Array, n_particles: int = 10_000, c0: jax.Array = CENTER, spread: float = 0.1) -> jax.Array:
    #generate symmetric starting data around the true center
    return c0 + spread*jax.random.normal(key, (n_particles, 2), dtype=jnp.float32)
    

def make_mu_grid(a_vals: jax.Array, omega_vals: jax.Array, d_vals: jax.Array) -> jax.Array:
    #single grid with all trials. Each row is a different phyiscal situation
    A, W, D = jnp.meshgrid(a_vals, omega_vals, d_vals, indexing="ij")
    grid = jnp.stack([A.ravel(),W.ravel(), D.ravel()], axis=-1)
    assert grid.shape[1]==dq, f"expected dq={dq}, god {grid.shape[1]}"
    return grid

def make_dataset(mu_grid: jax.Array, x0: jax.Array, key: jax.Array, sigma: float = 0.05, dt: float = 0.05, n_steps: int = 200, stride: int = 1):
    #mu_grid is the physical parameters for each dataset, x0 is the initial values, sigma, dt, n_steps are all for integration and stride is observation spacing
    #output is X = (N_mu, K+1, n_particles, 2) and t = (K+1,)

    keys = jax.random.split(key, mu_grid.shape[0]) #one key per mu

    def run_one(mu, k):
        k_int, k_shuf = jax.random.split(k) #keys for mu - two rands
        traj = euler_maruyama(lambda p: drift(p, mu), x0, k_int, sigma, dt, n_steps)
        traj = traj[::stride] #subsample in time
        return traj #returns POPULATION trajectory (each column is the population at stride but no indexing by particle)

    X = jax.vmap(run_one)(mu_grid, keys) #apply run to each set of parameters, mu
    t = jnp.arange(0, n_steps+1, stride) * dt #Not normalized time - literally t
    assert X.shape[1]==t.shape[0], f"time axis mismatch: X is {X.shape[1]} and t is {t.shape[0]}"
    return X, t

def sanity_check(X: jax.Array, t: jax.Array, mu_grid: jax.Array) -> None:
    #has the cloud actually relaxed onto the trough by the final time?
    for i, mu in enumerate(mu_grid):
        r_final = jnp.mean(jnp.linalg.norm(X[i, -1] - center(mu), axis=-1))
        r_trough = jnp.sqrt(mu[0])
        print(f"mu={jnp.round(mu,3)}  <r> = {r_final:.4f}  sqrt(a) = {r_trough:.4f}  ratio = {r_final/r_trough:.3f}")
 