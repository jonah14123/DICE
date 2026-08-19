"""
evaluation.py

Population-level comparison of the learned dynamics against the true dynamics,
following the measures used in the DICE paper (sections 8.3 - 8.5).
"""

import argparse
import json
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import serialization

from training import MLP, make_s
from make_dataset import make_dataset, make_init
from physics_integration import center

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn


### load a trained run
def load_run(tag):
    with open(f"{tag}/config_{tag}.json") as f:
        cfg = json.load(f)
    model = MLP(num_neuron=cfg["width"], num_layers=cfg["depth"], T=cfg["T"])
    template = model.init(jax.random.PRNGKey(0), jnp.zeros(2), jnp.zeros(1), jnp.zeros(3))
    with open(f"{tag}/params_{tag}.msgpack", "rb") as f:
        params = serialization.from_bytes(template, f.read())
    return model, params, cfg


def rebuild_data(cfg, mu_grid, n_x):
    #reproduce the true SDE data from the config alone, same key order as the driver
    key = jax.random.PRNGKey(cfg["seed"])
    key, k_x0, k_data, k_train = jax.random.split(key, 4)
    x0 = make_init(k_x0, n_particles=cfg["n_particles"], spread=cfg["spread"])[:n_x]
    x, t = make_dataset(mu_grid, x0, k_data, sigma=cfg["sigma"], dt=cfg["dt"],
                        n_steps=cfg["n_steps"], stride=cfg["stride"])
    return x, t, x0


### generate with the learned field: probability flow ODE, dX = grad s dt
def make_generator(s):
    grad_s = jax.grad(s, argnums=1)

    @jax.jit
    def generate(params, x0, t, mu):
        def step(carry, t_next):
            x, t_prev = carry
            dt = t_next - t_prev
            x_new = x + dt * jax.vmap(lambda p: grad_s(params, p, t_prev, mu))(x)
            return (x_new, t_next), x_new
        _, xs = jax.lax.scan(step, (x0, t[0]), t[1:])
        return jnp.concatenate([x0[None], xs], axis=0)
    return generate


### (1) moments, sec 8.3 fig 6.  radial analogue of <X^i> for this geometry
def moments(x, mu, orders=(1, 2, 3)):
    r = jnp.linalg.norm(x - center(mu), axis=-1)          #(K+1, N)
    m = jnp.stack([jnp.mean(r**i, axis=-1) for i in orders])
    sd = jnp.stack([jnp.std(r**i, axis=-1) for i in orders])
    return m, sd                                          #(n_ord, K+1)


### (4) kinetic energy, eq (78)
def kinetic_energy(x, t):
    d2 = jnp.sum((x[1:] - x[:-1])**2, axis=-1)            #(K, N)
    return float(jnp.sum(jnp.mean(d2, axis=-1) / (2*(t[1:] - t[:-1]))))


### (3) Sinkhorn divergence, sec 8.5
def sinkhorn_curve(x_true, x_gen, idx, eps, n_sub, key):
    def div(a, b):
        geom = pointcloud.PointCloud(a, b, epsilon=eps)
        out = sinkhorn.Sinkhorn()(linear_problem.LinearProblem(geom))
        return float(out.reg_ot_cost)  # type: ignore[arg-type]
    out = []
    for j in idx:
        k1, k2, key = jax.random.split(key, 3)
        a = jax.random.choice(k1, x_true[j], (n_sub,), replace=False)
        b = jax.random.choice(k2, x_gen[j],  (n_sub,), replace=False)
        out.append(float(div(a, b)))
    return np.array(out)


### plots
def plot_moments(tag, t, m_true, sd_true, m_gen, mu, label):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for i, ax in enumerate(axes):
        ax.plot(t, m_true[i], label="true")
        ax.fill_between(t, m_true[i] - sd_true[i]/10, m_true[i] + sd_true[i]/10, alpha=0.25)
        ax.plot(t, m_gen[i], "--", label="DICE")
        ax.set_xlabel("t"); ax.set_ylabel(f"<r^{i+1}>")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{label}  mu = ({mu[0]:.4g}, {mu[1]:.4g}, {mu[2]:.4g})", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{tag}/moments_{label}.png", dpi=150); plt.close(fig)


def plot_hist(tag, x_true, x_gen, t, idx, mu, label, bins=80):
    lo = float(jnp.minimum(x_true.min(), x_gen.min()))
    hi = float(jnp.maximum(x_true.max(), x_gen.max()))
    rng = [[lo, hi], [lo, hi]]
    fig, axes = plt.subplots(2, len(idx), figsize=(3*len(idx), 6.2))
    for c, j in enumerate(idx):
        for r, X in enumerate((x_true, x_gen)):
            H, _, _ = np.histogram2d(np.array(X[j][:, 0]), np.array(X[j][:, 1]),
                                     bins=bins, range=rng)
            axes[r, c].imshow(H.T, origin="lower", extent=[lo, hi, lo, hi], aspect="equal")
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[0, c].set_title(f"t = {float(t[j]):.1f}", fontsize=9)
    axes[0, 0].set_ylabel("true"); axes[1, 0].set_ylabel("DICE")
    fig.suptitle(f"{label}  mu = ({mu[0]:.4g}, {mu[1]:.4g}, {mu[2]:.4g})", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{tag}/hist_{label}.png", dpi=150); plt.close(fig)


def plot_sinkhorn(tag, t_sub, curves, labels):
    plt.figure(figsize=(6, 3.6))
    for c, l in zip(curves, labels):
        plt.plot(t_sub, c, marker="o", ms=3, label=l)
    plt.xlabel("t"); plt.ylabel("Sinkhorn divergence"); plt.yscale("log")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{tag}/sinkhorn_{tag}.png", dpi=150); plt.close()


### driver
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="population-level DICE evaluation")
    p.add_argument("--tag", type=str)
    args = p.parse_args()

    #NOTE Evaluation Params
    n_x = 2000            #particles used for evaluation
    n_sub = 1000          #particles per Sinkhorn call (cost is O(n^2))
    eps = 1e-3            #Sinkhorn entropic regularization
    t_stride = 20         #time subsampling for the Sinkhorn curve
    n_hist = 4            #snapshot times in the histogram figure
    seed = 0

    tag = args.tag
    model, params, cfg = load_run(tag)
    s = make_s(model)
    generate = make_generator(s)

    mu_train = jnp.array(cfg["mu_train"])
    mu_test  = jnp.array(cfg["mu_test"])
    mu_all   = jnp.concatenate([mu_train, mu_test])
    kind = ["train"]*len(mu_train) + ["test"]*len(mu_test)

    x_true_all, t, x0 = rebuild_data(cfg, mu_all, n_x)
    idx_s = list(range(0, len(t), t_stride))
    idx_h = list(np.linspace(0, len(t)-1, n_hist).astype(int))
    key = jax.random.PRNGKey(seed)

    print(f"{tag}: {len(mu_train)} train mu, {len(mu_test)} test mu, "
          f"n_x={n_x}, T={float(t[-1])}")
    print(f"{'kind':6s} {'a':>7s} {'omega':>7s} {'d':>7s} "
          f"{'sinkhorn':>10s} {'Ekin true':>10s} {'Ekin DICE':>10s}")

    curves, labels, rows = [], [], []
    for i, mu in enumerate(mu_all):
        x_true = x_true_all[i]
        x_gen = generate(params, x0, t, mu)

        key, k_s = jax.random.split(key)
        sk = sinkhorn_curve(x_true, x_gen, idx_s, eps, n_sub, k_s)
        e_true, e_gen = kinetic_energy(x_true, t), kinetic_energy(x_gen, t)
        rows.append([kind[i], *[float(v) for v in mu], float(sk.mean()), e_true, e_gen])
        print(f"{kind[i]:6s} {mu[0]:7.4f} {mu[1]:7.4f} {mu[2]:7.4f} "
              f"{sk.mean():10.5f} {e_true:10.3f} {e_gen:10.3f}")

        #figures for the first mu of each kind only
        if i == 0 or i == len(mu_train):
            label = f"{tag}_{kind[i]}"
            m_true, sd_true = moments(x_true, mu)
            m_gen, _ = moments(x_gen, mu)
            plot_moments(tag, t, m_true, sd_true, m_gen, mu, label)
            plot_hist(tag, x_true, x_gen, t, idx_h, mu, label)
            curves.append(sk); labels.append(label)

    plot_sinkhorn(tag, [float(t[j]) for j in idx_s], curves, labels)

    with open(f"{tag}/eval_{tag}.json", "w") as f:
        json.dump({"n_x": n_x, "n_sub": n_sub, "eps": eps, "rows": rows}, f, indent=2)

    tr = [r[4] for r in rows if r[0] == "train"]
    te = [r[4] for r in rows if r[0] == "test"]
    print(f"\nmean Sinkhorn   train {np.mean(tr):.5f}   test {np.mean(te):.5f}")
    print(f"wrote {tag}/moments_*.png  {tag}/hist_*.png  "
          f"{tag}/sinkhorn_{tag}.png  {tag}/eval_{tag}.json")