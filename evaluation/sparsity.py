import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from shared import align, support, expr_tree, tree_size

NEEDS_SIMULATION = False

# Aligns candidate and GT coefficients onto their shared set of terms
def _get(data):
    return align(data['lin_ref'], data['quad_ref'], data['lin_disc'], data['quad_disc'])

# Fraction of nonzero coefficients relative to the full candidate dictionary
def Sterms(data):
    _, c_disc, _ = _get(data)
    n_nonzero = int(np.sum(support(c_disc)))
    if n_nonzero == 0:
        return float('inf')
    theta_size = data.get('theta_size')
    if theta_size is None:
        raise ValueError("Sterms requires data['theta_size'] (size of the full candidate dictionary Theta) -- it cannot be inferred from a single ref/disc comparison.")
    return n_nonzero / theta_size

# Size (node count) of the equation's expression tree
def ExpTree(data):
    expr = expr_tree(data['b_disc'], data['lin_disc'], data['quad_disc'])
    return float(tree_size(expr))

METRICS = {'Sterms': Sterms, 'ExpTree': ExpTree}
