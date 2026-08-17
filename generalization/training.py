"""
train.py

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
    num_out: int = 1    #final dimension of output
    T: float = 10.0     #physical end time, for input normalization only

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

def train(x_data, t_data, mu_data, key, n_iter=20_000, bs_n=128, bs_t=64, bs_mu=5, width=128, depth=3, T=10.0):

    model = MLP(num_neuron=width, num_layers=depth, T=T)
    s = make_s(model)

    key, init_key = jax.random.split(key)
    schedule = make_schedule(n_iter)
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
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="parametric DICE training") #NOTE NOMENCLATURE: exp1, exp2, etc..
    p.add_argument("--tag", type=str, required=True, help="run name, used in output filenames")
    p.add_argument("--n_iter", type=int, default=2000, help="short run")
    args = p.parse_args()
    os.makedirs(args.tag, exist_ok=True)

    seed = 0
    key = jax.random.PRNGKey(seed)
    key, k_x0, k_data, k_train = jax.random.split(key, 4)

    #NOTE EXPERIMENT PARAMS
    a_vals     = jnp.array([0.05, 0.125, 0.20])
    omega_vals = jnp.array([0.00, 0.10, 0.20])
    d_vals     = jnp.array([0.25, 0.30, 0.35])
    mu_data = make_mu_grid(a_vals, omega_vals, d_vals)

    x0 = make_init(k_x0) #initial values/blob
    x_data, t_data = make_dataset(mu_data, x0, k_data)
    n_x=2000 #keep 2000 of 10000 particles per slice #NOTE FOR TIME
    x_data = x_data[:, :, :n_x, :]
    sanity_check(x_data, t_data, mu_data)
    t_data = t_data / 10.0 #normalize clock to 1.0

    width, depth, T, n_iter, bs_mu = 128, 3, 1.0, args.n_iter, 6
    #NOTE bs_mu changes per run!
    state, s, losses = train(x_data, t_data, mu_data, k_train, n_iter=n_iter, bs_n=128, bs_t=64, bs_mu=bs_mu, width=width, depth=depth, T=T) #NOTE change as needed for run time

    with open(f"{args.tag}/params_{args.tag}.msgpack", "wb") as f:
        f.write(serialization.to_bytes(state.params))
    np.save(f"{args.tag}/losses_{args.tag}.npy", losses)

    with open(f"{args.tag}/config_{args.tag}.json", "w") as f:
        json.dump({"width": width, "depth": depth, "T": T, "seed": seed,
                   "n_iter": args.n_iter, "n_x": n_x, "bs_mu": bs_mu,
                   "a_vals": a_vals.tolist(), "omega_vals": omega_vals.tolist(),
                   "d_vals": d_vals.tolist()}, f, indent=2)

    print(f"saved {args.tag} (width={width}, depth={depth}, T={T}, n_iter={args.n_iter})")
