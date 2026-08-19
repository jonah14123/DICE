"""
training.py

Parametric DICE training: minimize eq. (72) over theta.

TIME CONVENTION (changed -- now matches the paper):
    t_data is passed to the loss in PHYSICAL units, t in [0, T_phys].
    The paper does not normalize either: appendix B runs experiments on
    t in [0, 8], [0, 8.75], [0, 20], [0, 40].
    The MLP still divides its t INPUT by self.T for conditioning, but that is a
    feature scaling only -- it does not change the units of the output field.
    Consequence: the learned grad s_theta is the PHYSICAL probability-flow
    velocity u, directly comparable to -grad V. No t_scale factor anywhere.
"""
import argparse
import os
import json
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import serialization
from flax.training import train_state
from tqdm import tqdm

from diceLoss import get_dice_loss
from make_dataset import make_dataset, make_mu_grid, make_init, sanity_check

### network
class MLP(nn.Module):
    num_neuron: int     #neurons per layer
    num_layers: int     #depth
    num_out: int = 1    #final dimension of output
    T: float = 10.0     #physical end time, for INPUT normalization only

    def setup(self):
        self.layers = [nn.Dense(features=self.num_neuron) for _ in range(self.num_layers)]
        self.out = nn.Dense(features=self.num_out)

    def __call__(self, x, t, mu):
        h = jnp.hstack([x, t/self.T, mu]) #t/T is a feature scaling, not a change of units
        for layer in self.layers:
            h = nn.swish(layer(h))
        h = self.out(h)
        return h

def init_state(key, model, schedule):
    params = model.init(key, jnp.zeros(2), jnp.zeros(1), jnp.zeros(3))
    optimizer = optax.adam(learning_rate=schedule)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

def make_schedule(n_iter, peak_lr=5e-4, final_lr=1e-6):
    return optax.cosine_decay_schedule(
        init_value=peak_lr,
        decay_steps=n_iter,
        alpha=final_lr / peak_lr)      # alpha is a MULTIPLIER, not a floor

### Physics
def make_s(model): #scalar field
    def s(params, x, t, mu):
        return model.apply(params, x, t, mu).sum()
    return s

### training
def make_train_step(loss_fn):
    @jax.jit
    def train_step(state, key):
        loss, grads = jax.value_and_grad(loss_fn)(state.params, key)
        return state.apply_gradients(grads=grads), loss
    return train_step

def train(x_data, t_data, mu_data, key, n_iter=20_000, bs_n=256, bs_t=128, bs_mu=1,
          width=128, depth=7, T=10.0, peak_lr=5e-4, final_lr=1e-6):

    model = MLP(num_neuron=width, num_layers=depth, T=T)
    s = make_s(model)

    key, init_key = jax.random.split(key)
    schedule = make_schedule(n_iter, peak_lr, final_lr)
    state = init_state(init_key, model, schedule)

    loss_fn = get_dice_loss(s, x_data, t_data, mu_data, bs_n, bs_t, bs_mu)
    train_step = make_train_step(loss_fn)

    losses = []
    with tqdm(range(n_iter)) as pbar:
        for it in pbar:
            key, sub = jax.random.split(key)      # FRESH key every iteration
            state, loss = train_step(state, sub)
            losses.append(loss)
            if it % 100 == 0:
                pbar.set_postfix({"loss": float(loss)})

    return state, s, np.array(losses)

### driver
EXPERIMENTS = {
    1: dict(a=[0.05, 0.125, 0.20], omega=[0.0],             d=[0.0],
            holdout_a=[0.125], holdout_omega=[], holdout_d=[]),
    2: dict(a=[0.05, 0.125, 0.20], omega=[0.0, 0.10, 0.20], d=[0.0],
            holdout_a=[0.125], holdout_omega=[], holdout_d=[]),
    3: dict(a=[0.05, 0.125, 0.20], omega=[0.0, 0.10, 0.20], d=[0.25, 0.30, 0.35],
            holdout_a=[0.125], holdout_omega=[], holdout_d=[]),
}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="parametric DICE training")
    p.add_argument("--n_iter", type=int, default=20_000)
    p.add_argument("--tag", type=str)
    p.add_argument("--exp", type=int)
    p.add_argument("--seed", type=int, default=0)
 
    args = p.parse_args()
 
    #NOTE Network Params
    width = 128
    depth = 7
    bs_n = 256
    bs_t = 128
    bs_mu = 1
    peak_lr = 5e-4
    final_lr = 1e-6
 
    #NOTE Physical Params
    sigma = 0.05
    dt = 0.05
    n_steps = 200
    stride = 1
    spread = 0.1
    n_particles = 10_000
    n_x = 10_000
 
    #Make folder
    tag = args.tag
    os.makedirs(tag, exist_ok=True)
 
    #assign random keys
    key = jax.random.PRNGKey(args.seed)
    key, k_x0, k_data, k_train = jax.random.split(key, 4)
 
    #choose mu - held-out values are EXCLUDED from training, kept for testing.
    grid = EXPERIMENTS[args.exp]
    a_train = [a for a in grid["a"]     if a not in grid["holdout_a"]]
    w_train = [w for w in grid["omega"] if w not in grid["holdout_omega"]]
    d_train = [d for d in grid["d"]     if d not in grid["holdout_d"]]
    if not (a_train and w_train and d_train):
        raise ValueError("holdout removed every value along some axis")
 
    rows = lambda A, W, D: [[float(a), float(w), float(d)]
                            for a in A for w in W for d in D]
    mu_train = rows(a_train, w_train, d_train)
    mu_test  = [row for row in rows(grid["a"], grid["omega"], grid["d"])
                if row not in mu_train]
 
    mu_data = jnp.array(mu_train)
    n_mu = mu_data.shape[0]
    if bs_mu > n_mu:
        raise ValueError(f"bs_mu={bs_mu} > n_mu={n_mu}")
    if n_x > n_particles:
        raise ValueError(f"n_x={n_x} > n_particles={n_particles}")
 
    #total time
    T_phys = n_steps * dt
 
    #make initial particles at t=0
    x0 = make_init(k_x0, n_particles=n_particles, spread=spread)
    x_data, t_data = make_dataset(mu_data, x0, k_data, sigma=sigma, dt=dt,
                                  n_steps=n_steps, stride=stride)
    x_data = x_data[:, :, :n_x, :]
    sanity_check(x_data, t_data, mu_data)
 
    gb = x_data.size * 4 / 1e9
    print(f"{tag}: n_mu={n_mu}  x_data={x_data.shape} ({gb:.2f} GB)  T_phys={T_phys}  sigma={sigma}")
    print(f"  held out for testing: a={grid['holdout_a']} omega={grid['holdout_omega']} "
          f"d={grid['holdout_d']}  ({len(mu_test)} mu)")
 
    state, s, losses = train(x_data, t_data, mu_data, k_train,
                             n_iter=args.n_iter, bs_n=bs_n, bs_t=bs_t,
                             bs_mu=bs_mu, width=width, depth=depth,
                             T=T_phys, peak_lr=peak_lr, final_lr=final_lr)
 
    with open(f"{tag}/params_{tag}.msgpack", "wb") as f:
        f.write(serialization.to_bytes(state.params))
    np.save(f"{tag}/losses_{tag}.npy", losses)
 
    cfg = {
        "exp": args.exp, "seed": args.seed, "time_convention": "physical",
        "width": width, "depth": depth, "T": T_phys,
        "n_iter": args.n_iter, "bs_n": bs_n, "bs_t": bs_t, "bs_mu": bs_mu,
        "peak_lr": peak_lr, "final_lr": final_lr,
        "sigma": sigma, "dt": dt, "n_steps": n_steps, "stride": stride,
        "spread": spread, "n_particles": n_particles, "n_x": n_x,
        "a_vals": grid["a"], "omega_vals": grid["omega"], "d_vals": grid["d"],
        "holdout_a": grid["holdout_a"], "holdout_omega": grid["holdout_omega"],
        "holdout_d": grid["holdout_d"],
        "mu_train": mu_train, "mu_test": mu_test,
    }
    with open(f"{tag}/config_{tag}.json", "w") as f:
        json.dump(cfg, f, indent=2)
 
    print(f"saved {tag}  (width={width}, depth={depth}, bs_mu={bs_mu}, "
          f"n_iter={args.n_iter}, final loss={losses[-1]:.6f})")
