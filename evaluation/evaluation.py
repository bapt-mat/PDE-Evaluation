import argparse
import glob
import importlib.util
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import SIMULATORS, PDE_ROOT, parse_pde_file, pde_from_filename, pde_label, make_integrator, simulate

parser = argparse.ArgumentParser()
parser.add_argument('--file', required=True, nargs='+')
parser.add_argument('--ref', default=None)
parser.add_argument('--metric', required=True, nargs='+')
parser.add_argument('--dt_factor', type=float, default=1.0)
parser.add_argument('--T_max', type=float, default=[None], nargs='+', help='One or more simulation end times; repeats the full run for each')
parser.add_argument('--k_max', type=float, default=None, help='Max spatial wavenumber for fMSE (default: 1%%-of-peak-energy auto cutoff)')
parser.add_argument('--theta_size', type=int, default=None, help='Size of the full candidate dictionary Theta, required by Sterms')
parser.add_argument('--baseline_file', default=None, help='PDE file used as the baseline model for Score (default: null model u_t=0)')
parser.add_argument('--ic', type=int, default=None, help='OOD initial condition choice for IC_MAE: 1 (default/training IC), 2 (Gaussian bump), 3 (random Fourier modes)')
parser.add_argument('--n_refine', type=int, default=None, help='Number of successive halvings/doublings for Conv_t/Conv_x (default: 6)')
parser.add_argument('--c0', type=float, default=None, help='Reward1 sparsity/accuracy trade-off (default: 0.2)')
parser.add_argument('--xi1', type=float, default=None, help='Reward2 sparsity weight (default: 0.01)')
parser.add_argument('--xi2', type=float, default=None, help='Reward2 tree-depth weight (default: 0.0001)')
parser.add_argument('--rank', type=float, default=None, help='MDL_Fey candidate rank N(alpha_hat) (default: 1, no penalty)')
parser.add_argument('--mdl_lambda', type=float, default=None, help='MDL_Fey trade-off lambda (default: sqrt(n data points))')
parser.add_argument('--eps_d', type=float, default=None, help='MDL_Fey normalization constant (default: 1e-15)')
parser.add_argument('--c_max', type=float, default=None, help='MDL_Sym max constant magnitude (default: 10.0)')
parser.add_argument('--eps_enc', type=float, default=None, help='MDL_Sym constant encoding precision (default: 1e-6)')
parser.add_argument('--mdl_sym_lambda', type=float, default=None, help='MDL_Sym trade-off lambda (default: 0.1)')
args = parser.parse_args()

HERE = os.path.dirname(__file__)
registry = {}
by_file = {}
for fname in sorted(os.listdir(HERE)):
    if not fname.endswith('.py') or fname in ('evaluation.py', 'utils.py'):
        continue
    stem = fname[:-3]
    path = os.path.join(HERE, fname)
    spec = importlib.util.spec_from_file_location(stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, 'METRICS'):
        continue
    needs_sim = getattr(mod, 'NEEDS_SIMULATION', True)
    for name, fn in mod.METRICS.items():
        registry[name] = fn, needs_sim
        by_file.setdefault(stem, []).append(name)

if args.metric == ['all']:
    args.metric = sorted(registry)
else:
    expanded = []
    for m in args.metric:
        expanded.extend(sorted(by_file[m]) if m in by_file else [m])
    args.metric = expanded

unknown = [m for m in args.metric if m not in registry]
if unknown:
    print(f'Unknown metric(s): {unknown}. Available:\n  ' + '\n  '.join(sorted(registry)) + '\n  Or a file name to select all its metrics:\n  ' + '\n  '.join(sorted(by_file)))
    sys.exit(1)

needs_simulation = any(registry[m][1] for m in args.metric)

if len(args.file) == 1 and not args.file[0].endswith('.txt'):
    prefix = args.file[0]
    pde_dir = os.path.join(PDE_ROOT, prefix)
    all_files = sorted(glob.glob(os.path.join(pde_dir, f'{prefix}_*.txt')))
    ref_path = os.path.join(pde_dir, f'{prefix}_pde_true.txt')
    assert ref_path in all_files, f'Reference file not found: {ref_path}'
    compare_files = [f for f in all_files if f != ref_path]
else:

    # Resolves a bare filename to its full path under pde_files/
    def _resolve(path):
        if os.path.isabs(path) or os.path.exists(path):
            return path
        d = os.path.join(PDE_ROOT, pde_from_filename(path))
        return os.path.join(d, os.path.basename(path))
    resolved = [_resolve(p) for p in args.file]
    pde_name = pde_from_filename(resolved[0])
    pde_dir = os.path.join(PDE_ROOT, pde_name)
    ref_path = args.ref if args.ref else os.path.join(pde_dir, f'{pde_name}_pde_true.txt')
    compare_files = resolved

pde_name = pde_from_filename(ref_path)

for T_max in args.T_max:
    if len(args.T_max) > 1:
        print(f"\n{'=' * 20} T_max={T_max} {'=' * 20}")
    sim_kwargs = {'T': T_max} if T_max is not None else {}
    x, t, fields = SIMULATORS[pde_name](**sim_kwargs)
    u0 = fields['u'][:, 0].copy()
    N = len(x)
    nt = len(t)
    nt_inner = max(1, round(1.0 / args.dt_factor))
    dt = float(t[1] - t[0]) / nt_inner
    L = float(x[-1] - x[0] + (x[1] - x[0]))
    kk = np.fft.rfftfreq(N, d=L / N) * 2 * np.pi

    print(f'Simulating reference: {ref_path}')
    b_r, lin_r, quad_r, xlin_r = parse_pde_file(ref_path)
    U_ref = simulate(make_integrator(b_r, lin_r, quad_r, kk, dt, x_coord=x, x_lin=xlin_r), u0, nt, nt_inner)
    mean_abs_ref = float(np.mean(np.abs(U_ref)))

    baseline = None
    if args.baseline_file:
        bl_path = args.baseline_file
        if not (os.path.isabs(bl_path) or os.path.exists(bl_path)):
            bl_path = os.path.join(PDE_ROOT, pde_from_filename(bl_path), os.path.basename(bl_path))
        print(f'Simulating baseline (Score): {bl_path}')
        b_bl, lin_bl, quad_bl, xlin_bl = parse_pde_file(bl_path)
        U_bl = simulate(make_integrator(b_bl, lin_bl, quad_bl, kk, dt, x_coord=x, x_lin=xlin_bl), u0, nt, nt_inner)
        baseline = {'b_base': b_bl, 'lin_base': lin_bl, 'quad_base': quad_bl, 'xlin_base': xlin_bl, 'U_base': U_bl}

    print(f"\nMetrics: {', '.join(args.metric)}\n" + '-' * 40)
    for fpath in compare_files:
        b, lin, quad, xlin = parse_pde_file(fpath)
        print(f'\nEvaluating: {os.path.basename(fpath)}')
        print(f'  PDE: u_t = {pde_label(b, lin, quad, x_lin=xlin)}')

        U_disc = simulate(make_integrator(b, lin, quad, kk, dt, x_coord=x, x_lin=xlin), u0, nt, nt_inner) if needs_simulation else None
        data = {'U_ref': U_ref, 'U_disc': U_disc, 'x': x, 't': t, 'kk': kk, 'dt': dt, 'mean_abs_ref': mean_abs_ref, 'k_max': args.k_max, 'theta_size': args.theta_size, 'b_ref': b_r, 'lin_ref': lin_r, 'quad_ref': quad_r, 'xlin_ref': xlin_r, 'b_disc': b, 'lin_disc': lin, 'quad_disc': quad, 'xlin_disc': xlin}
        if baseline is not None:
            data.update(baseline)
        for key in ('c0', 'xi1', 'xi2', 'rank', 'mdl_lambda', 'eps_d', 'c_max', 'eps_enc', 'mdl_sym_lambda', 'ic', 'n_refine'):
            val = getattr(args, key)
            if val is not None:
                data[key] = val

        for m in args.metric:
            compute, _ = registry[m]
            try:
                result = compute(data)
            except ValueError:
                print(f'  {m}: argument missing')
                continue
            if isinstance(result, dict):
                print(f'  {m}:')
                for k, v in result.items():
                    print(f'    {k}: {v}')
            else:
                print(f'  {m}: {result}')
