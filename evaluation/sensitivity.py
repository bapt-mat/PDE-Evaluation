import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np


# Builds a JAX-differentiable ETD1 trajectory function for given PDE terms
def _trajectory_fn(lin_keys, quad_keys, kk, dt, N, b, x_coord, xlin_items, nt, nt_inner):
    kk = jnp.asarray(kk)
    xlin_items = tuple(xlin_items)
    if x_coord is not None:
        x_coord = jnp.asarray(x_coord)
    needed_orders = tuple(sorted(set(lin_keys) | {oi for oi, _ in quad_keys} | {oj for _, oj in quad_keys} | {o for o, _ in xlin_items}))

    def derivs(u):
        uh = jnp.fft.rfft(u)
        return {o: u if o == 0 else jnp.fft.irfft((1j * kk) ** o * uh, n=N) for o in needed_orders}

    def run(lin_vals, quad_vals, u0):
        L_hat = sum(c * (1j * kk) ** o for o, c in zip(lin_keys, lin_vals)).astype(complex) if lin_keys else jnp.zeros_like(kk, dtype=complex)
        exp_L = jnp.exp(L_hat * dt)
        safe_L = jnp.where(jnp.abs(L_hat) < 1e-14, 1.0, L_hat)
        coef_L = jnp.where(jnp.abs(L_hat) < 1e-14, dt, (exp_L - 1.0) / safe_L)

        def step(u):
            d = derivs(u)
            nl = jnp.full((N,), b, dtype=jnp.float64)
            for ((oi, oj), val) in zip(quad_keys, quad_vals):
                nl = nl + val * d[oi] * d[oj]
            for o, c in xlin_items:
                nl = nl + c * x_coord * d[o]
            return jnp.fft.irfft(exp_L * jnp.fft.rfft(u) + coef_L * jnp.fft.rfft(nl), n=N)

        def inner(u, _):
            return step(u), None

        def outer(u, _):
            u, _ = jax.lax.scan(inner, u, None, length=nt_inner)
            return u, u
        _, U_rest = jax.lax.scan(outer, u0, None, length=nt - 1)
        return jnp.concatenate([u0[None, :], U_rest], axis=0)
    return run

# Sum-of-squares sensitivity dU/dalpha_i for each term, via forward-mode autodiff
def sensitivity_weights(data):
    lin_ref, quad_ref = data['lin_ref'], data['quad_ref']
    lin_disc, quad_disc = data['lin_disc'], data['quad_disc']
    lin_keys = tuple(sorted(set(lin_ref) | set(lin_disc)))
    quad_keys = tuple(sorted(set(((oi, oj) for oi, oj, _ in quad_ref)) | set(((oi, oj) for oi, oj, _ in quad_disc))))

    def _qval(quad, key):
        for oi, oj, v in quad:
            if (oi, oj) == key:
                return v
        return 0.0
    lin_vals = jnp.array([lin_ref.get(k, 0.0) for k in lin_keys], dtype=jnp.float64)
    quad_vals = jnp.array([_qval(quad_ref, k) for k in quad_keys], dtype=jnp.float64)
    U_ref = data['U_ref']
    N, nt = U_ref.shape
    dt = float(data['dt'])
    t = data['t']
    nt_inner = max(1, round(float(t[1] - t[0]) / dt))
    u0 = jnp.asarray(U_ref[:, 0], dtype=jnp.float64)
    b = float(data.get('b_ref', 0.0))
    xlin_items = tuple(sorted(data.get('xlin_ref', {}).items()))
    traj_fn = _trajectory_fn(lin_keys, quad_keys, data['kk'], dt, N, b, data.get('x'), xlin_items, nt, nt_inner)

    def f(lin_vals, quad_vals):
        return traj_fn(lin_vals, quad_vals, u0)
    jac_lin, jac_quad = jax.jacfwd(f, argnums=(0, 1))(lin_vals, quad_vals)
    w_lin = np.array(jnp.sum(jac_lin ** 2, axis=(0, 1)))
    w_quad = np.array(jnp.sum(jac_quad ** 2, axis=(0, 1)))
    return lin_keys, quad_keys, w_lin, w_quad
