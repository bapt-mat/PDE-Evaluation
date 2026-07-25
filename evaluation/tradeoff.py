import os, sys
import numpy as np
import sympy as sp
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared import align, support, expr_tree, tree_depth, tree_size
from utils import eval_rhs, make_integrator, simulate

NEEDS_SIMULATION = False

# Aligns candidate and GT coefficients onto their shared set of terms
def _get(data):
    return align(data['lin_ref'], data['quad_ref'], data['lin_disc'], data['quad_disc'])

# Evaluates GT vs. candidate RHS pointwise on the reference solution
def _residual_ut(data):
    ut_true = eval_rhs(data['b_ref'], data['lin_ref'], data['quad_ref'], data['U_ref'], data['kk'])
    ut_hat = eval_rhs(data['b_disc'], data['lin_disc'], data['quad_disc'], data['U_ref'], data['kk'])
    return ut_true, ut_hat

# Normalized mean absolute error between two fields
def _nMAE(U_disc, U_ref):
    num = float(np.sum(np.abs(U_disc - U_ref)))
    den = float(np.sum(np.abs(U_ref)))
    return num / den if den > 0 else float('nan')

# Accuracy-per-complexity gain of the candidate vs. a baseline PDE
def Score(data):
    if 'lin_base' not in data:
        raise ValueError("Score requires a baseline PDE (data['lin_base']/['quad_base']/['b_base'], e.g. via evaluation.py's --baseline_file) -- it has no meaningful default.")
    U_ref = data['U_ref']
    u0 = U_ref[:, 0]
    b, lin, quad = data['b_disc'], data['lin_disc'], data['quad_disc']
    U_disc = data.get('U_disc')
    if U_disc is None:
        dt = float(data['dt'])
        integrator = make_integrator(b, lin, quad, data['kk'], dt, x_coord=data.get('x'), x_lin=data.get('xlin_disc', {}))
        U_disc = simulate(integrator, u0, U_ref.shape[1])
    b_base, lin_base, quad_base = data['b_base'], data['lin_base'], data['quad_base']
    U_base = data.get('U_base')
    if U_base is None:
        dt = float(data['dt'])
        integrator = make_integrator(b_base, lin_base, quad_base, data['kk'], dt, x_coord=data.get('x'), x_lin=data.get('xlin_base', {}))
        U_base = simulate(integrator, u0, U_ref.shape[1])
    nmae_disc = _nMAE(U_disc, U_ref)
    nmae_base = _nMAE(U_base, U_ref)
    C_disc = tree_size(expr_tree(b, lin, quad))
    C_base = tree_size(expr_tree(b_base, lin_base, quad_base))
    dC = C_disc - C_base
    if dC == 0 or nmae_disc <= 0 or nmae_base <= 0:
        return float('nan')
    return float(-(np.log(nmae_disc) - np.log(nmae_base)) / dC)

# Sparsity-weighted R^2 fit on the u_t residual
def Reward1(data):
    c0 = data.get('c0', 0.2)
    _, c_disc, _ = _get(data)
    s_size = int(np.sum(support(c_disc)))
    if s_size == 0:
        return float('-inf')
    sparsity = 1 - c0 * np.log10(s_size)
    ut_true, ut_hat = _residual_ut(data)
    rss = float(np.sum((ut_true - ut_hat) ** 2))
    tss = float(np.sum((ut_true - ut_true.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else float('nan')
    return float(sparsity * r2)

# Sparsity/depth-penalized inverse residual error
def Reward2(data):
    xi1 = data.get('xi1', 0.01)
    xi2 = data.get('xi2', 0.0001)
    _, c_disc, _ = _get(data)
    s_size = int(np.sum(support(c_disc)))
    depth = tree_depth(expr_tree(data['b_disc'], data['lin_disc'], data['quad_disc']))
    ut_true, ut_hat = _residual_ut(data)
    err = float(np.sum((ut_true - ut_hat) ** 2))
    return float((1 - xi1 * s_size - xi2 * depth) / (1 + err))

# Corrected Akaike Information Criterion (fit vs. sparsity trade-off)
def AICc(data):
    _, c_disc, _ = _get(data)
    s_size = int(np.sum(support(c_disc)))
    ut_true, ut_hat = _residual_ut(data)
    n = ut_true.size
    rss = float(np.sum((ut_true - ut_hat) ** 2))
    return float(n * np.log(rss / n) + 2 * s_size)

# Bayesian Information Criterion (fit vs. sparsity trade-off)
def BIC(data):
    _, c_disc, _ = _get(data)
    s_size = int(np.sum(support(c_disc)))
    ut_true, ut_hat = _residual_ut(data)
    n = ut_true.size
    rss = float(np.sum((ut_true - ut_hat) ** 2))
    return float(np.log(n) * s_size + n * np.log(rss / n))

# AI-Feynman-style minimum description length score
def MDL_Fey(data):
    rank = data.get('rank', 1)
    eps_d = data.get('eps_d', 1e-15)
    ut_true, ut_hat = _residual_ut(data)
    n = ut_true.size
    lam = data.get('mdl_lambda', np.sqrt(n))
    eps = float(np.mean((ut_true - ut_hat) ** 2))
    return float(np.log2(rank) + lam * np.log2(max(1.0, eps / eps_d)))

# Description-length term (len(e)) used by MDL_Sym
def _description_length(expr, data, theta_size):
    c_max = data.get('c_max', 10.0)
    eps = data.get('eps_enc', 1e-06)
    bits = 0.0
    for node in sp.preorder_traversal(expr):
        if isinstance(node, (sp.Add, sp.Mul)):
            bits += np.log2(len(node.args))
    bits += theta_size * np.log2(c_max / eps)
    return bits

# SymLang-style minimum description length score
def MDL_Sym(data):
    lam = data.get('mdl_sym_lambda', 0.1)
    _, c_disc, _ = _get(data)
    theta_size = data.get('theta_size') or len(c_disc)
    ut_true, ut_hat = _residual_ut(data)
    n = ut_true.size
    rss = float(np.sum((ut_true - ut_hat) ** 2))
    neg_log_p = 0.5 * n * np.log(rss / n)
    expr = expr_tree(data['b_disc'], data['lin_disc'], data['quad_disc'])
    length = _description_length(expr, data, theta_size)
    return float(neg_log_p + lam * length)

METRICS = {'Score': Score, 'MDL_Sym': MDL_Sym, 'Reward1': Reward1, 'Reward2': Reward2, 'AICc': AICc, 'BIC': BIC, 'MDL_Fey': MDL_Fey}
