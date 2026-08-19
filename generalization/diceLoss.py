"""
dice_loss.py

Fully empirical, parametric DICE loss: eq. (70) of the paper summed over mu,
which is eq. (72). Adapted from the demo notebook to the array layout produced
by make_dataset: x_data of shape (N_mu, K+1, N_x, d)

"""
import jax
import jax.numpy as jnp

def random_subset_of(key, arr, bs): #get random subset of an array of size bs
    N = len(arr)
    indices = jax.random.choice(key, N, shape=(bs,), replace=False)
    subset = arr[indices]
    return subset

def random_timegrid_of(key, t_data, bs_t):
    t_q = random_subset_of(key, t_data, bs_t) #draw bs_t observation times for training
    t_q = jnp.sort(t_q) #sort to make sure adjacent observations are adjacent
    t_q = t_q.at[0].set(t_data[0]) #pin the boundary points (initial and final)
    t_q = t_q.at[-1].set(t_data[-1])

    w_q = 0.5 * jnp.concatenate([jnp.array([t_q[1] - t_q[0]]), (t_q[2:] - t_q[:-2]), jnp.array([t_q[-1] - t_q[-2]])]) #trapazoidal weights
    return t_q, w_q

def get_expected_value_fct(x_data, t_data, mu_data, bs_n): #returns a function for expected value
    n_t = t_data.shape[0]
    def expected_value(f, tau, t, i_mu, key): #computes E[f(x,t,mu)] ~ p(tau;mu)
        #rho is evaluated at tau, s is evaluated at t
        #invert the uniform time grid; round, do not truncate
        i_t = jnp.round((tau - t_data[0]) / (t_data[-1] - t_data[0]) * (n_t - 1)).astype(jnp.int32)
        x = random_subset_of(key, x_data[i_mu, i_t], bs_n) #(N_mu, K+1, N_x, d) layout
        mu = mu_data[i_mu]
        #signature of f should be (x, t, mu) -> scalar
        return jnp.mean(jax.vmap(lambda _x: f(_x, t, mu))(x))
    return expected_value


def get_s_derivatives(s): #gradiant function and gradiant squared - as opposed to time derivatives in AM
    def grad_s(x, t, mu):
        return jax.grad(s)(x, t, mu)
    def grad_s_squared(x, t, mu):
        return jnp.sum(grad_s(x, t, mu)**2)
    return grad_s, grad_s_squared

### DICE

def get_dice_loss(_s, x_data, t_data, mu_data, bs_n, bs_t, bs_mu):  #return dice_loss function
    n_mu = mu_data.shape[0]
    expected_value = get_expected_value_fct(x_data, t_data, mu_data, bs_n) #define expected value function

    def dice_loss_mu(params, key, i_mu): #EQ 70 for one mu

        #closures for s
        s = lambda x, t, mu: _s(params, x, t, mu)
        grad_s, grad_s_squared = get_s_derivatives(s)

        key, x_key, t_key = jax.random.split(key, 3)
        x_keys = jax.random.split(x_key, bs_t)
        t_q, w_q = random_timegrid_of(t_key, t_data, bs_t)

        E_s = lambda tau, t, key: expected_value(s, tau, t, i_mu, key)
        E_s_v = jax.vmap(E_s)

        sum_En_snplus1 =      jnp.sum(E_s_v(t_q[:-1], t_q[1:],  x_keys[:-1])) #second term of Eq 70 expanded over linear expectation
        sum_Enplus1_sn =      jnp.sum(E_s_v(t_q[1:],  t_q[:-1], x_keys[1:]))
        sum_En_sn =           jnp.sum(E_s_v(t_q[:-1], t_q[:-1], x_keys[:-1]))
        sum_Enplus1_snplus1 = jnp.sum(E_s_v(t_q[1:],  t_q[1:],  x_keys[1:]))
        loss = (+ 0.5 * sum_En_snplus1
                - 0.5 * sum_Enplus1_sn
                + 0.5 * sum_En_sn
                - 0.5 * sum_Enplus1_snplus1)

        E_grad_s_squared = lambda tau, t, key: expected_value(grad_s_squared, tau, t, i_mu, key) #kinetic term - trapazoid integration approximation
        loss += 0.5 * jnp.sum( w_q * jax.vmap(E_grad_s_squared)(t_q, t_q, x_keys))

        return loss

    def dice_loss(params, key): #EQ 72
        key, mu_key = jax.random.split(key)
        i_mus = jax.random.choice(mu_key, n_mu, shape=(bs_mu,), replace=False)
        mu_keys = jax.random.split(key, bs_mu) #one key PER mu
        #minibatch sum has expectation (bs_mu / n_mu) * L_mu
        loss = jnp.sum(jax.vmap(lambda i_mu, k: dice_loss_mu(params, k, i_mu))(i_mus, mu_keys))
        return loss * (n_mu/bs_mu) #when bs_mu == n_mu the factor is 1 and this is the literal eq 72

    return dice_loss