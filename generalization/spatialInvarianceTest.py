"""
Proposition 3: L_DICE(s + f) = L_DICE(s) for any spatially-constant, time-varying f
"""
import jax
import jax.numpy as jnp

from diceLoss import get_dice_loss

def s_base(params, x, t, mu):
    return jnp.sum(params * x) * jnp.exp(-t) + mu[0] * jnp.sum(x**2) + mu[1] * x[0] * x[1]

def f(t):
    #spatially constant, time varying -- the thing s is determined only up to
    return jnp.sin(3.0 * t) + 2.0 * t**2

def s_shifted(params, x, t, mu):
    return s_base(params, x, t, mu) + f(t)

def s_broken(params, x, t, mu):
    #NEGATIVE CONTROL: an f that is NOT spatially constant should break invariance
    return s_base(params, x, t, mu) + jnp.sin(3.0 * t) * x[0]

### fake data with the make_dataset layout (N_mu, K+1, N_x, d)
key = jax.random.PRNGKey(0)
k_x, k_mu = jax.random.split(key)

N_mu, K1, N_x = 4, 21, 500
x_data = jax.random.normal(k_x, (N_mu, K1, N_x, 2))
t_data = jnp.linspace(0.0, 10.0, K1)          #physical time, uniform grid
mu_data = jax.random.uniform(k_mu, (N_mu, 3))

bs_n, bs_t, bs_mu = 64, 8, 2
params = jnp.array([1.0, -0.5])

#DICE loss for each
L_base    = get_dice_loss(s_base,    x_data, t_data, mu_data, bs_n, bs_t, bs_mu)
L_shifted = get_dice_loss(s_shifted, x_data, t_data, mu_data, bs_n, bs_t, bs_mu)
L_broken  = get_dice_loss(s_broken,  x_data, t_data, mu_data, bs_n, bs_t, bs_mu)

### SAME key for all three -> identical time grids and identical particle subsets
eval_key = jax.random.PRNGKey(42)
l_base    = L_base(params, eval_key)
l_shifted = L_shifted(params, eval_key)
l_broken  = L_broken(params, eval_key)

rel = lambda a, b: abs(a - b) / max(abs(a), 1e-12)

print(f"L(s)         = {l_base: .8f}")
print(f"L(s + f(t))  = {l_shifted: .8f}   rel diff = {rel(l_base, l_shifted):.2e}")
print(f"L(s + f(t)x) = {l_broken: .8f}   rel diff = {rel(l_base, l_broken):.2e}")
print("\nrel diff should sit at machine epsilon (~1e-7 in float32, ~1e-15 in float64)\n ")