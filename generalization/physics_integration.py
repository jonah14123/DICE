"""
Here we define potential and drift functions tht depend on x and mu:

potential(x, mu) -> scalar V(x; mu) for a single point x in two space
drift(x, mu) -> -delV, the jax.gradoant of potential in addition to some rotationalDrift

mu convention (fixed, do not change):
    mu[0] = a       squared trough radius
    mu[1] = omega   rotation rate about the center, positive = clockwise
    mu[2] = d       displacement of the center along x, so c = C + [d, 0]


The overall goal is to use the paramaterized dice loss in section 7.2 of the DICE paper and see if/how it generalizes.
"""
import jax.numpy as jnp
import jax
from collections.abc import Callable
from functools import partial #to bake mu into drift

CENTER = jnp.array([0.5,0.5]) #base center of circle
dq = 3 #dimensions of mu

def center(mu: jax.Array, c: jax.Array = CENTER) -> jax.Array:
    #actual center c(mu) = C + [d, 0]
    assert mu.shape == (dq,), f"expected dq={dq}, got {mu.shape}"
    return c + jnp.array([mu[2], 0.0])

def potential(p: jax.Array, mu: jax.Array) -> jax.Array: #curl-free
    #double well V = (|p-c|^2 - a)^2
    assert mu.shape == (dq,), f"expected dq={dq}, got {mu.shape}"

    r2 = jnp.sum((p-center(mu))**2) #sum over all components and square
    a = mu[0]
    return (r2-a)**2

def rotationalDrift(p: jax.Array, mu: jax.Array) -> jax.Array: #divergence free
    #strict rotation D = [y, -x] about c(mu), so positive omega is clockwise
    assert mu.shape == (dq,), f"expected dq={dq}, got {mu.shape}"
    J_rot = jnp.array([[0.0,1.0],[-1.0,0.0]])
    return J_rot @ (p-center(mu))

def drift(p: jax.Array, mu: jax.Array) -> jax.Array:
    #drift = -del V + omega * rotationalDrift
    assert mu.shape == (dq,), f"expected dq={dq}, got {mu.shape}"
    omega = mu[1]
    return -jax.grad(potential)(p, mu) + omega*rotationalDrift(p, mu)

def make_drift(mu: jax.Array) -> Callable[[jax.Array], jax.Array]:
    #bake mu in to get the one-argument callable
    return partial(drift, mu=mu)

"""
Now we define the integration function we will use. This function is made independent of the physics and depends on the drift function
euler_maruyama(x0, sigma, dt, n_steps, key) -> array(n_steps+1, n_particles, d)
"""
def euler_maruyama(drift_fn: Callable[[jax.Array], jax.Array], x0: jax.Array, key: jax.Array, sigma: float = 0.05, dt: float = 0.05, n_steps: int = 200) -> jax.Array:
    #Euler-Maruyama with dX = drift_fn(X) dt + sigma dW
    #x0 = (n_particles, d) and is sample from rho(0)
    #sigma is the random noise at each step
    keys = jax.random.split(key, n_steps)
    drift_batched = jax.vmap(drift_fn)

    def step(x, k):
        dW = sigma * jax.random.normal(k, x.shape, dtype=x.dtype) * jnp.sqrt(dt)
        x_next = x + dt * drift_batched(x) + dW
        return x_next, x_next #new carry, output to stack

    _, traj = jax.lax.scan(step, x0, keys) #iterate steps/integrate
    return jnp.concatenate([x0[None],traj], axis=0) #add x0 to begining