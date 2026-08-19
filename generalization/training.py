"""
traiing.py

Parametric DICE training: minimize eq. (72) over theta
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
    T: float            #physical end time
    num_out: int = 1    #final dimension of output
    
    def setup(self):
        self.layers = [nn.Dense(features=self.num_neuron) for _ in range(self.num_layers)] #initializes layers with num_hid features
        self.out = nn.Dense(features=self.num_out) #creates final layer that outputs num_out features

    def __call__(self, x, t, mu):
        h = jnp.hstack([x, t/self.T, mu]) #normalize t here
        for layer in self.layers:
            h = nn.swish(layer(h)) #applies linear transformation layer on h over all layers
        h = self.out(h) #applies final linear transformation to get output
        return h

def init_state(key, model, schedule): #begin with intiital del s_theta (noise)
    params = model.init(key, jnp.zeros(2), jnp.zeros(1), jnp.zeros(3))
    #The three dummy arguments are shape probes
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

def train(x_data, t_data, mu_data, key, n_iter=20_000, bs_n=256, bs_t=128, bs_mu=1,  width=128, depth=7, T=10.0, peak_lr=5e-4, final_lr=1e-6):

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
    1: dict(a=[0.05, 0.125, 0.20], omega=[0.0],             d=[0.0]),
    2: dict(a=[0.05, 0.125, 0.20], omega=[0.0, 0.10, 0.20], d=[0.0]),
    3: dict(a=[0.05, 0.125, 0.20], omega=[0.0, 0.10, 0.20], d=[0.25, 0.30, 0.35]),
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

    #choose mu
    grid = EXPERIMENTS[args.exp]
    mu_data = make_mu_grid(jnp.array(grid["a"]), jnp.array(grid["omega"]), jnp.array(grid["d"]))
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
 
    state, s, losses = train(x_data, t_data, mu_data, k_train,
                             n_iter=args.n_iter, bs_n=bs_n, bs_t=bs_t,
                             bs_mu=bs_mu, width=width, depth=depth,
                             T=T_phys, peak_lr=peak_lr, final_lr=final_lr)
 
    with open(f"{tag}/params_{tag}.msgpack", "wb") as f:
        f.write(serialization.to_bytes(state.params))
    np.save(f"{tag}/losses_{tag}.npy", losses)
 
    cfg = {
        "exp": args.exp, "seed": args.seed,
        "width": width, "depth": depth, "T": T_phys,
        "n_iter": args.n_iter, "bs_n": bs_n, "bs_t": bs_t, "bs_mu": bs_mu,
        "peak_lr": peak_lr, "final_lr": final_lr,
        "sigma": sigma, "dt": dt, "n_steps": n_steps, "stride": stride,
        "spread": spread, "n_particles": n_particles, "n_x": n_x,
        "a_vals": grid["a"], "omega_vals": grid["omega"], "d_vals": grid["d"],
    }
    with open(f"{tag}/config_{tag}.json", "w") as f:
        json.dump(cfg, f, indent=2)
 
    print(f"saved {tag}  (width={args.width}, depth={args.depth}, bs_mu={args.bs_mu}, "
          f"n_iter={args.n_iter}, final loss={losses[-1]:.6f})")
