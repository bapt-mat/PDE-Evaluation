import os, sys
import numpy as np
import sympy as sp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import ORDER_NAME

THRESH = 1e-10

# Builds ref/disc coefficient vectors over the union of terms they use
def align(lin_ref, quad_ref, lin_disc, quad_disc):
    lin_keys = sorted(set(lin_ref) | set(lin_disc))
    quad_keys = sorted(set(((oi, oj) for oi, oj, _ in quad_ref)) | set(((oi, oj) for oi, oj, _ in quad_disc)))

    def _qval(quad, key):
        for oi, oj, v in quad:
            if (oi, oj) == key:
                return v
        return 0.0

    c_ref = np.array([lin_ref.get(k, 0.0) for k in lin_keys] + [_qval(quad_ref, k) for k in quad_keys])
    c_disc = np.array([lin_disc.get(k, 0.0) for k in lin_keys] + [_qval(quad_disc, k) for k in quad_keys])
    labels = [f"u_{('x' * k if k > 0 else '')}" for k in lin_keys] + [f"u_{'x' * oi}·u_{'x' * oj}" for oi, oj in quad_keys]
    return c_ref, c_disc, labels

# Boolean mask of nonzero coefficients
def support(c, thresh=THRESH):
    return np.abs(c) > thresh

# sympy symbol for a given derivative order (u, u_x, u_xx, ...)
def symbol(o):
    return sp.Symbol(ORDER_NAME.get(o, 'u_' + 'x' * o))

# Builds a sympy expression for the PDE's right-hand side
def expr_tree(b, linear, quad):
    terms = []
    if abs(b) > 1e-12:
        terms.append(sp.Float(b))
    for o, c in linear.items():
        if abs(c) > 1e-12:
            terms.append(c * symbol(o))
    for oi, oj, v in quad:
        if abs(v) > 1e-12:
            terms.append(v * symbol(oi) * symbol(oj))
    return sp.Add(*terms) if terms else sp.Integer(0)

# Number of nodes in an expression tree
def tree_size(expr):
    return len(list(sp.preorder_traversal(expr)))

# Depth of an expression tree
def tree_depth(expr):
    if not expr.args:
        return 1
    return 1 + max(tree_depth(a) for a in expr.args)
