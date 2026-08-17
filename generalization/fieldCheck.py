import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
 
from evaluation import load_run
from training import make_s
from make_dataset import make_mu_grid
from physics_integration import center
 
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--a", type=float, required=True, help="use a TRAINED value")
    p.add_argument("--omega", type=float, default=0.0)
    p.add_argument("--d", type=float, default=0.0)
    args = p.parse_args()
 
    model, params, cfg = load_run(args.tag)
    s = make_s(model)
    grad_s = jax.grad(s, argnums=1) #params is arg 0, x is arg 1
 
    mu = make_mu_grid(jnp.array([args.a]), jnp.array([args.omega]), jnp.array([args.d]))[0]
    c = center(mu)
    r_trough = float(jnp.sqrt(mu[0]))
 
    #probe along the +x ray from the well center
    r = jnp.linspace(0.01, 0.8, 200)
    pts = c + jnp.stack([r, jnp.zeros_like(r)], axis=-1)
    e_r = jnp.stack([jnp.ones_like(r), jnp.zeros_like(r)], axis=-1) #radial unit vector on this ray
 
    plt.figure(figsize=(7, 4.5))
    for t in [0.5, 2.0, 5.0, 9.0]:
        g = jax.vmap(lambda q: grad_s(params, q, t, mu))(pts)
        g_r = jnp.sum(g * e_r, axis=-1) #radial component
        plt.plot(r, g_r, label=f"grad s_theta, t={t}")
 
    plt.plot(r, -4*r*(r**2 - mu[0]), "k--", lw=1, label="-V'(r)  [scale reference]")
    plt.axvline(r_trough, color="gray", lw=0.8)
    plt.axhline(0.0, color="gray", lw=0.8)
    plt.text(r_trough, plt.ylim()[1]*0.9, r" $\sqrt{a}$", color="gray")
    plt.xlabel("r = |p - c|"); plt.ylabel("radial component")
    plt.title(f"{args.tag}:  mu = ({args.a}, {args.omega}, {args.d})")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{args.tag}/field_{args.tag}_a{args.a}.png", dpi=150)
    print(f"wrote {args.tag}/field_{args.tag}_a{args.a}.png")
 
    #printed summary
    print(f"\nsqrt(a) = {r_trough:.4f}")
    for t in [0.5, 2.0, 5.0, 9.0]:
        g = jax.vmap(lambda q: grad_s(params, q, t, mu))(pts)
        g_r = jnp.sum(g * e_r, axis=-1)
        i_cross = jnp.argmin(jnp.abs(g_r))
        print(f"  t={t:4.1f}   max|g_r| = {jnp.max(jnp.abs(g_r)):.4f}   "
              f"g_r nearest zero at r = {r[i_cross]:.4f}")
    print(f"\n  reference: max|V'| on this range = {jnp.max(jnp.abs(4*r*(r**2-mu[0]))):.4f}")