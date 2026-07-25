import os
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
SIM_CACHEDIR = os.path.join(DIR, 'sim_cache')
PDE_ROOT = os.path.join(DIR, 'pde_files')
ORDER_NAME = {0: 'u', 1: 'u_x', 2: 'u_xx', 3: 'u_xxx', 4: 'u_xxxx'}

# Spatial derivative of order p via FFT
def spectral_deriv(U, k, p):
    N, nt = U.shape
    return np.array([np.fft.irfft((1j * k) ** p * np.fft.rfft(U[:, n]), n=N) for n in range(nt)]).T

# Reference simulation of the viscous Burgers' equation
def simulate_burgers(nu=0.05):
    N = 512
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    k = np.fft.rfftfreq(N, d=L / N) * 2 * np.pi
    dt = 0.0025
    nt = 800
    t = np.linspace(0, nt * dt, nt)
    u0 = np.sin(x) + 0.5 * np.sin(2 * x) + 0.3 * np.sin(3 * x)
    lin = -nu * k ** 2
    exp_l = np.exp(lin * dt)
    safe = np.where(np.abs(lin) < 1e-14, 1.0, lin)
    coef = np.where(np.abs(lin) < 1e-14, dt, (exp_l - 1.0) / safe)

    U = np.zeros((N, nt))
    U[:, 0] = u0
    u = u0.copy()

    print('Simulating Burgers...')
    for n in range(nt - 1):
        u_hat = np.fft.rfft(u)
        u_hat = exp_l * u_hat + coef * (-0.5j * k * np.fft.rfft(u ** 2))
        u = np.fft.irfft(u_hat, n=N)
        U[:, n + 1] = u
    return x, t, {'u': U, 'u_x': spectral_deriv(U, k, 1), 'u_xx': spectral_deriv(U, k, 2)}

# Reference simulation of the Korteweg-de Vries equation
def simulate_kdv():
    N = 512
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    k = np.fft.rfftfreq(N, d=L / N) * 2 * np.pi
    dt = 1.0 / (1200 - 1)
    nt = 1200
    t = np.linspace(0, 1.0, nt)
    u0 = 0.5 * np.cos(x) + 0.3 * np.cos(2 * x + 0.5)
    lin = 1j * k ** 3
    exp_l = np.exp(lin * dt)
    safe = np.where(np.abs(lin) < 1e-14, 1.0, lin)
    coef = np.where(np.abs(lin) < 1e-14, dt, (exp_l - 1.0) / safe)

    U = np.zeros((N, nt))
    U[:, 0] = u0
    u = u0.copy()

    print('Simulating KdV...')
    for n in range(nt - 1):
        u_hat = np.fft.rfft(u)
        u_hat = exp_l * u_hat + coef * (-6 * 0.5j * k * np.fft.rfft(u ** 2))
        u = np.fft.irfft(u_hat, n=N)
        U[:, n + 1] = u
    return x, t, {'u': U, 'u_x': spectral_deriv(U, k, 1), 'u_xx': spectral_deriv(U, k, 2)}

# Reference simulation of the Kuramoto-Sivashinsky equation
def simulate_ks(T=50.0):
    N = 1024
    L = 22.0
    x = np.linspace(0, L, N, endpoint=False)
    k = np.fft.rfftfreq(N, d=L / N) * 2 * np.pi
    dt = 0.05
    nt = max(1000, round(T / dt) + 1)
    t = np.arange(nt) * dt
    u0 = np.cos(2 * np.pi * x / L) + 0.5 * np.cos(4 * np.pi * x / L + 0.3)
    lin = k ** 2 - k ** 4
    exp_l = np.exp(lin * dt)
    safe = np.where(np.abs(lin) < 1e-14, 1.0, lin)
    coef = np.where(np.abs(lin) < 1e-14, dt, (exp_l - 1.0) / safe)
    dealias = np.abs(k) < 2 / 3 * k.max()

    U = np.zeros((N, nt))
    U[:, 0] = u0
    u = u0.copy()

    print('Simulating KS...')
    for n in range(nt - 1):
        u_hat = np.fft.rfft(u)
        nl = 1j * k * np.fft.rfft(-0.5 * u ** 2) * dealias
        u_hat = exp_l * u_hat + coef * nl
        u = np.fft.irfft(u_hat, n=N)
        U[:, n + 1] = u
    return x, t, {'u': U, 'u_x': spectral_deriv(U, k, 1), 'u_xx': spectral_deriv(U, k, 2)}

# Wraps a simulator so its output is cached to disk between runs
def cached_simulator(name, sim_fn):

    # Loads the cached simulation if present, otherwise runs and caches it
    def wrapper(*args, **kwargs):
        os.makedirs(SIM_CACHEDIR, exist_ok=True)
        suffix = '_'.join((f'{k}{v}' for k, v in sorted(kwargs.items()))) if kwargs else ''
        cache_file = os.path.join(SIM_CACHEDIR, f"{name}{('_' + suffix if suffix else '')}.npz")
        if os.path.exists(cache_file):
            print(f'Loading cached {name} simulation from {cache_file}...')
            data = np.load(cache_file)
            x = data['x']
            t = data['t']
            fields = {k: data[k] for k in data.files if k not in ('x', 't')}
            return x, t, fields
        x, t, fields = sim_fn(*args, **kwargs)
        np.savez(cache_file, x=x, t=t, **fields)
        print(f'Cached {name} simulation → {cache_file}')
        return x, t, fields
    return wrapper
SIMULATORS = {'burgers': cached_simulator('burgers', simulate_burgers), 'kdv': cached_simulator('kdv', simulate_kdv), 'ks': cached_simulator('ks', simulate_ks)}

# Derivative order encoded in a term name (number of 'x' characters)
def deriv_order(name):
    return name.count('x')

# Parses a PDE coefficient .txt file into (bias, linear, quadratic, x-linear) terms
def parse_pde_file(path):
    b = 0.0
    linear = {}
    quad = []
    x_lin = {}

    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#') or ':' not in s:
                continue
            key, val_str = s.split(':', 1)
            key = key.strip()
            if key in ('pde', 'base', 'source'):
                continue
            try:
                val = float(val_str.strip())
            except ValueError:
                continue
            if key in ('b', 'bias'):
                b = val
            elif '·' in key:
                parts = [p.strip() for p in key.split('·')]
                is_x = lambda p: p in ('x', 'X')
                if is_x(parts[0]):
                    o = deriv_order(parts[1])
                    x_lin[o] = x_lin.get(o, 0.0) + val
                elif is_x(parts[1]):
                    o = deriv_order(parts[0])
                    x_lin[o] = x_lin.get(o, 0.0) + val
                else:
                    quad.append((deriv_order(parts[0]), deriv_order(parts[1]), val))
            else:
                o = deriv_order(key)
                linear[o] = linear.get(o, 0.0) + val
    return b, linear, quad, x_lin

# Extracts the PDE name (e.g. 'ks') from a filename
def pde_from_filename(path):
    return os.path.basename(path).split('_')[0]

# Human-readable string of a PDE's terms and coefficients
def pde_label(b, linear, quad, sep=',  ', x_lin=None):
    parts = []
    if abs(b) > 1e-12:
        parts.append(f'b={b:.3g}')
    for o in sorted(linear):
        parts.append(f"{ORDER_NAME.get(o, 'u_' + 'x' * o)}={linear[o]:.3g}")
    for oi, oj, val in quad:
        ni = ORDER_NAME.get(oi, 'u_' + 'x' * oi)
        nj = ORDER_NAME.get(oj, 'u_' + 'x' * oj)
        parts.append(f'{ni}·{nj}={val:.3g}')
    for o in sorted(x_lin or {}):
        nu = ORDER_NAME.get(o, 'u_' + 'x' * o)
        parts.append(f'x·{nu}={x_lin[o]:.3g}')
    return sep.join(parts)

# Builds the ETD1 time-stepping function for a given PDE
def make_integrator(b, linear, quad, kk, dt, x_coord=None, x_lin=None):
    L_hat = sum(c * (1j * kk) ** o for o, c in linear.items()).astype(complex)
    exp_L = np.exp(L_hat * dt)
    safe_L = np.where(np.abs(L_hat) < 1e-14, 1.0, L_hat)
    coef_L = np.where(np.abs(L_hat) < 1e-14, dt, (exp_L - 1.0) / safe_L)
    N = len(kk) * 2 - 2
    cache = {}
    x_lin = x_lin or {}

    # Cached spatial derivative of order o, for the current step
    def d(u, o):
        if o not in cache:
            uh = np.fft.rfft(u)
            cache[o] = u if o == 0 else np.fft.irfft((1j * kk) ** o * uh, n=N)
        return cache[o]

    # Advances the solution by one ETD1 time step
    def step(u):
        cache.clear()
        nl = np.full(N, b, dtype=float)
        for oi, oj, val in quad:
            nl += val * d(u, oi) * d(u, oj)
        for o, c in x_lin.items():
            nl += c * x_coord * d(u, o)
        return np.fft.irfft(exp_L * np.fft.rfft(u) + coef_L * np.fft.rfft(nl), n=N)
    return step

# Runs the time-stepping loop, storing the solution at each output step
def simulate(step_fn, u0, nt, nt_inner=1):
    N = len(u0)
    U = np.zeros((N, nt))
    u = u0.copy()
    U[:, 0] = u
    for n in range(nt - 1):
        for _ in range(nt_inner):
            u = step_fn(u)
        U[:, n + 1] = u
    return U

# Flattens linear/quadratic coefficients into a single vector
def coeff_vector(linear, quad):
    return np.array([linear[o] for o in sorted(linear)] + [v for _, _, v in quad])

# Returns a cached accessor for spatial derivatives of a field
def get_derivs(U, kk):
    N, nt = U.shape
    cache = {0: U}

    # Computes (and caches) the derivative of order o
    def get(o):
        if o not in cache:
            cache[o] = np.array([np.fft.irfft((1j * kk) ** o * np.fft.rfft(U[:, n]), n=N) for n in range(nt)]).T
        return cache[o]
    return get

# Evaluates a PDE's right-hand side pointwise on a given field
def eval_rhs(b, linear, quad, U, kk):
    get = get_derivs(U, kk)
    rhs = np.full(U.shape, b, dtype=float)
    for o, c in linear.items():
        rhs += c * get(o)
    for oi, oj, val in quad:
        rhs += val * get(oi) * get(oj)
    return rhs

# Average magnitude of each term, evaluated on the reference solution
def term_weights(linear, quad, U_ref, kk):
    get = get_derivs(U_ref, kk)
    w_lin = {o: float(np.mean(np.abs(get(o)))) for o in linear}
    w_quad = [(oi, oj, float(np.mean(np.abs(get(oi) * get(oj))))) for oi, oj, _ in quad]
    return w_lin, w_quad
