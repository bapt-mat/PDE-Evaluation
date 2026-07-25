import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from errors import rollout
from utils import make_integrator, simulate

NEEDS_SIMULATION = True

# Default/training initial condition
def _ic1(x, L):
    return np.cos(2 * np.pi * x / L) + 0.5 * np.cos(4 * np.pi * x / L + 0.3)

# OOD initial condition: two sine modes at a generic relative phase
def _ic2(x, L):
    return np.sin(2 * np.pi * x / L) + 0.3 * np.sin(6 * np.pi * x / L + 0.5)

# OOD initial condition: two cosine modes at a generic relative phase
def _ic3(x, L):
    return np.cos(6 * np.pi * x / L) + 0.2 * np.cos(10 * np.pi * x / L + 0.7)

IC_FUNCS = {1: _ic1, 2: _ic2, 3: _ic3}

# Resimulates GT and the candidate PDE from a chosen initial condition
def _simulate_ic(data, ic_choice):
    if ic_choice not in IC_FUNCS:
        raise ValueError(f'Unknown IC choice {ic_choice!r}; must be 1, 2, or 3.')
    x, kk, dt = data['x'], data['kk'], data['dt']
    L = float(x[-1] - x[0] + (x[1] - x[0]))
    u0 = IC_FUNCS[ic_choice](x, L)
    nt = data['U_ref'].shape[1]
    ref_int = make_integrator(data['b_ref'], data['lin_ref'], data['quad_ref'], kk, dt, x_coord=x, x_lin=data.get('xlin_ref', {}))
    U_ref_ic = simulate(ref_int, u0, nt)
    disc_int = make_integrator(data['b_disc'], data['lin_disc'], data['quad_disc'], kk, dt, x_coord=x, x_lin=data.get('xlin_disc', {}))
    U_disc_ic = simulate(disc_int, u0, nt)
    return u0, U_ref_ic, U_disc_ic

# nMAE between candidate and GT under an unseen initial condition
def IC_nMAE(data):
    ic_choice = data.get('ic', 1)
    _, U_ref_ic, U_disc_ic = _simulate_ic(data, ic_choice)
    num = float(np.sum(np.abs(U_disc_ic - U_ref_ic)))
    den = float(np.sum(np.abs(U_ref_ic)))
    return num / den if den > 0 else float('nan')

# Rollout error evaluated over a longer (OOD) horizon
def rollout_OOD(data):
    return rollout(data)

# Simulates the same PDE at successively halved time steps
def conv_t_series(b, lin, quad, x, kk, u0, T, dt0, n_refine, xlin=None):
    dts = [dt0 / 2 ** i for i in range(n_refine + 1)]
    finals = []
    for dt_i in dts:
        nt_i = max(2, round(T / dt_i) + 1)
        integrator = make_integrator(b, lin, quad, kk, dt_i, x_coord=x, x_lin=xlin or {})
        finals.append(simulate(integrator, u0, nt_i)[:, -1])
    diffs = []
    for i in range(len(finals) - 1):
        num = float(np.sum(np.abs(finals[i + 1] - finals[i])))
        den = float(np.sum(np.abs(finals[i + 1])))
        diffs.append(num / den if den > 0 else float('nan'))
    return dts, diffs

# Simulates the same PDE at successively doubled spatial resolutions
def conv_x_series(b, lin, quad, x0, dt, T, ic_choice, n_refine, xlin=None):
    L = float(x0[-1] - x0[0] + (x0[1] - x0[0]))
    Ns = [len(x0) * 2 ** i for i in range(n_refine + 1)]
    finals = []
    for N_i in Ns:
        x_i = np.linspace(0, L, N_i, endpoint=False)
        kk_i = np.fft.rfftfreq(N_i, d=L / N_i) * 2 * np.pi
        u0_i = IC_FUNCS[ic_choice](x_i, L)
        nt_i = max(2, round(T / dt) + 1)
        integrator = make_integrator(b, lin, quad, kk_i, dt, x_coord=x_i, x_lin=xlin or {})
        finals.append(simulate(integrator, u0_i, nt_i)[:, -1])
    diffs = []
    for i in range(len(finals) - 1):
        fine_sub = finals[i + 1][::2]
        num = float(np.sum(np.abs(fine_sub - finals[i])))
        den = float(np.sum(np.abs(fine_sub)))
        diffs.append(num / den if den > 0 else float('nan'))
    return Ns, diffs

# Checks numerical convergence of the candidate PDE as the time step shrinks
def Conv_t(data):
    b, lin, quad = data['b_disc'], data['lin_disc'], data['quad_disc']
    x, kk = data['x'], data['kk']
    u0 = data['U_ref'][:, 0]
    T = float(data['t'][-1])
    dt0 = float(data['dt'])
    n_refine = int(data.get('n_refine', 6))
    dts, diffs = conv_t_series(b, lin, quad, x, kk, u0, T, dt0, n_refine, data.get('xlin_disc'))
    return {f'dt={dts[i]:.2g}->dt={dts[i + 1]:.2g}': diffs[i] for i in range(len(diffs))}

# Checks numerical convergence of the candidate PDE as the spatial grid is refined
def Conv_x(data):
    b, lin, quad = data['b_disc'], data['lin_disc'], data['quad_disc']
    x0 = data['x']
    dt = float(data['dt'])
    T = float(data['t'][-1])
    ic_choice = data.get('ic', 1)
    n_refine = int(data.get('n_refine', 6))
    Ns, diffs = conv_x_series(b, lin, quad, x0, dt, T, ic_choice, n_refine, data.get('xlin_disc'))
    return {f'N={Ns[i]}->N={Ns[i + 1]}': diffs[i] for i in range(len(diffs))}

METRICS = {'IC_nMAE': IC_nMAE, 'rollout_OOD': rollout_OOD, 'Conv_t': Conv_t, 'Conv_x': Conv_x}
