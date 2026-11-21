import numpy as np
def equation(t, A, params):
    p = params
    g = [np.ones_like(t), t, t**2, t**3]
    f = [np.ones_like(A), A, A**2]
    out = 0
    k = 0
    for fa in f:
        for gt in g:
            out = out + p[k]*fa*gt
            k += 1
    return out
