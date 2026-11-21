import numpy as np
def equation(eps, T, params):
    p=params
    funcs=[np.ones_like(eps), np.log(eps), eps, 1/eps, eps**2]
    out=0
    k=0
    for f in funcs:
        for q in [0,1]:
            out = out + p[k]*f*T**q
            k+=1
    return out
