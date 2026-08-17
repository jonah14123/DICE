import argparse
import json
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import serialization

from training import MLP, make_s
from make_dataset import make_dataset, make_mu_grid, make_init
from physics_integration import center

### load
def load_run(tag):
    with open(f"{tag}/config_{tag}.json") as f:
        cfg = json.load(f)
    model = MLP(num_neuron=cfg["width"], num_layers=cfg["depth"], T=cfg["T"])
    template = model.init(jax.random.PRNGKey(0), jnp.zeros(2), jnp.zeros(1), jnp.zeros(3))
    with open(f"{tag}/params_{tag}.msgpack", "rb") as f:
        params = serialization.from_bytes(template, f.read())
    return model, params, cfg

def get_t_scale(cfg): #RESCALE
    return float(cfg.get("t_scale", 10.0))

### generate with the learned field
def make_generator(s):
    grad_s = jax.grad(s, argnums=1) #params is arg 0 here, x is arg 1

    def generate(params, x0, t_grid, mu):
        #explicit euler on the probability-flow ODE; t_grid must be in NETWORK time
        def step(carry, t_next):
            x, t_prev = carry
            dt = t_next - t_prev
            x_new = x + dt * jax.vmap(lambda p: grad_s(params, p, t_prev, mu))(x)
            return (x_new, t_next), x_new

        (_, _), xs = jax.lax.scan(step, (x0, t_grid[0]), t_grid[1:])
        return jnp.concatenate([x0[None], xs], axis=0) #(K+1, n_x, 2)
    return generate

### diagnostics
def mean_radius(x, mu):
    #<|p - c(mu)|> at each time, shape (K+1,)
    return jnp.mean(jnp.linalg.norm(x - center(mu), axis=-1), axis=-1)

def plot_comparison(x_true, x_gen, t_phys, mu, idx, fname):
    fig, axes = plt.subplots(2, len(idx), figsize=(3.2*len(idx), 6.4), sharex=True, sharey=True)
    for col, j in enumerate(idx):
        axes[0, col].scatter(x_true[j,:,0], x_true[j,:,1], s=0.5, alpha=0.3)
        axes[1, col].scatter(x_gen[j,:,0],  x_gen[j,:,1],  s=0.5, alpha=0.3, color="C1")
        axes[0, col].set_title(f"t = {t_phys[j]:.1f}")
        for row in (0, 1):
            axes[row, col].set_aspect("equal")
    axes[0,0].set_ylabel("true (SDE)")
    axes[1,0].set_ylabel("generated (grad s)")
    fig.suptitle(f"held-out mu = ({mu[0]:.4g}, {mu[1]:.4g}, {mu[2]:.4g})")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    print(f"wrote {fname}")

def sample_test(vals, override, key):
    #uniform in [min, max] of the training grid
    if override is not None:
        return float(override)
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return float(lo)
    return float(jax.random.uniform(key, minval=lo, maxval=hi))

#field and loss tests
def plot_loss(tag, window=50):
    #smoothed DICE loss curve; raw curve is jagged from mu-to-mu variation
    L = np.load(f"{tag}/losses_{tag}.npy")
    plt.figure(figsize=(6, 4))
    plt.plot(np.convolve(L, np.ones(window)/window, mode="valid"))
    plt.xlabel("iteration"); plt.ylabel(f"DICE loss (smoothed, w={window})")
    plt.title(tag); plt.tight_layout()
    plt.savefig(f"{tag}/loss_{tag}.png", dpi=150)
    print(f"wrote {tag}/loss_{tag}.png   (L[0]={L[0]:.5f}  L[-1]={L[-1]:.5f})")

def plot_field(tag, params, s, mu, t_scale, t_net=(0.05, 0.2, 0.5, 0.9)):
    grad_s = jax.grad(s, argnums=1) #params is arg 0, x is arg 1
    c = center(mu)
    r_trough = float(jnp.sqrt(mu[0]))
 
    r = jnp.linspace(0.01, 0.8, 200)
    pts = c + jnp.stack([r, jnp.zeros_like(r)], axis=-1)
    e_r = jnp.stack([jnp.ones_like(r), jnp.zeros_like(r)], axis=-1) #radial unit vector on +x ray
 
    ref = -t_scale * 4*r*(r**2 - mu[0]) #scale reference in normalized-time units
 
    plt.figure(figsize=(7, 4.5))
    print(f"\nfield check: mu = ({mu[0]:.4g}, {mu[1]:.4g}, {mu[2]:.4g})   sqrt(a) = {r_trough:.4f}")
    for t in t_net:
        g = jax.vmap(lambda q: grad_s(params, q, t, mu))(pts)
        g_r = jnp.sum(g * e_r, axis=-1)
        plt.plot(r, g_r, label=f"grad s_theta, t_net={t}  (t_phys={t*t_scale:.1f})")
        i0 = jnp.argmin(jnp.abs(g_r))
        print(f"  t_net={t:4.2f}   max|g_r| = {jnp.max(jnp.abs(g_r)):8.4f}   "
              f"|g_r| min at r = {r[i0]:.4f}")
 
    plt.plot(r, ref, "k--", lw=1, label=f"-{t_scale:g} V'(r)  [scale reference]")
    plt.axvline(r_trough, color="gray", lw=0.8)
    plt.axhline(0.0, color="gray", lw=0.8)
    plt.text(r_trough, plt.ylim()[1]*0.9, r" $\sqrt{a}$", color="gray")
    plt.xlabel("r = |p - c|"); plt.ylabel("radial component (normalized-time units)")
    plt.title(f"{tag}:  mu = ({mu[0]:.4g}, {mu[1]:.4g}, {mu[2]:.4g})")
    plt.legend(fontsize=8); plt.tight_layout()
 
    fname = f"{tag}/field_{tag}_a{mu[0]:.4g}_w{mu[1]:.4g}_d{mu[2]:.4g}.png"
    plt.savefig(fname, dpi=150)
    print(f"  reference max|{t_scale:g} V'| = {jnp.max(jnp.abs(ref)):.4f}")
    print(f"wrote {fname}")

### driver

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--test_seed", type=int, default=1) #which random test point
    p.add_argument("--a_test", type=float, default=None)
    p.add_argument("--omega_test", type=float, default=None)
    p.add_argument("--d_test", type=float, default=None)
    p.add_argument("--field_a", type=float, default=None,
                   help="probe the field at a TRAINED a instead of the test point")
    p.add_argument("--no_sample", action="store_true",
                   help="skip the ODE sampling check, diagnostics only")
    args = p.parse_args()

    model, params, cfg = load_run(args.tag)
    s = make_s(model)
    generate = make_generator(s)
    t_scale = get_t_scale(cfg)

    #test mu drawn uniformly from the training range of THIS grid
    k_a, k_w, k_d = jax.random.split(jax.random.PRNGKey(args.test_seed), 3)
    a_t = sample_test(cfg["a_vals"],     args.a_test,     k_a)
    w_t = sample_test(cfg["omega_vals"], args.omega_test, k_w)
    d_t = sample_test(cfg["d_vals"],     args.d_test,     k_d)
    mu_test = make_mu_grid(jnp.array([a_t]), jnp.array([w_t]), jnp.array([d_t]))
    mu = mu_test[0]
    tail = f"a{a_t:.4g}_w{w_t:.4g}_d{d_t:.4g}"

    print(f"\nheld-out mu = ({a_t:.4g}, {w_t:.4g}, {d_t:.4g})   sqrt(a) = {jnp.sqrt(mu[0]):.4f}")
    print(f"t_scale = {t_scale}   nearest trained a = "
          f"{min(cfg['a_vals'], key=lambda v: abs(v - a_t)):.4g}")

    ### 1. loss curve -- did it converge?
    plot_loss(args.tag)

    ### 2. field check -- is grad s_theta the right shape and magnitude?
    #    default to a TRAINED mu so a bad result cannot be blamed on generalization
    if args.field_a is not None:
        mu_field = make_mu_grid(jnp.array([args.field_a]),
                                jnp.array([cfg["omega_vals"][0]]),
                                jnp.array([cfg["d_vals"][0]]))[0]
    else:
        mu_field = mu
    plot_field(args.tag, params, s, mu_field, t_scale)

    ### 3. sampling check -- does the learned field transport the population?
    if not args.no_sample:
        key = jax.random.PRNGKey(cfg["seed"])
        key, k_x0, k_data, k_train = jax.random.split(key, 4)
        n_x = cfg["n_x"]
        x0 = make_init(k_x0)[:n_x] #match the training particle count

        #physics is unchanged: make_dataset returns PHYSICAL time
        x_true, t_phys = make_dataset(mu_test, x0, k_data)
        x_true = x_true[0] #(K+1, n_x, 2)

        #the network lives on the normalized clock, so integrate there
        t_net = t_phys / t_scale
        x_gen = generate(params, x0, t_net, mu)

        r_true, r_gen = mean_radius(x_true, mu), mean_radius(x_gen, mu)
        print(f"\nmean radius (physical t in [0, {float(t_phys[-1]):.3g}])")
        for j in [0, len(t_phys)//4, len(t_phys)//2, -1]:
            print(f"  t={t_phys[j]:5.2f}   true {r_true[j]:.4f}   gen {r_gen[j]:.4f}   "
                  f"err {abs(r_true[j]-r_gen[j]):.4f}")

        idx = [0, len(t_phys)//3, 2*len(t_phys)//3, len(t_phys)-1]
        plot_comparison(x_true, x_gen, t_phys, mu, idx,
                        f"{args.tag}/compare_{args.tag}_{tail}.png")

        plt.figure(figsize=(5,3.5))
        plt.plot(t_phys, r_true, label="true")
        plt.plot(t_phys, r_gen, "--", label="generated")
        plt.axhline(float(jnp.sqrt(mu[0])), color="k", lw=0.5, label="sqrt(a)")
        plt.xlabel("physical t"); plt.ylabel("<|p - c|>"); plt.legend(); plt.tight_layout()
        plt.savefig(f"{args.tag}/radius_{args.tag}_{tail}.png", dpi=150)
        print(f"wrote {args.tag}/radius_{args.tag}_{tail}.png")