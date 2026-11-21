import numpy as np
def equation(eps, T, params):
    p = np.asarray(params)
    L = np.log(T)
    G = [np.ones_like(T), L, 1.0/T, T, T**2]
    out = 0.0
    k = 0
    for i in range(4):
        for g in G:
            out = out + p[k]*(eps**i)*g
            k += 1
    return out
