import numpy as np
def equation(eps, T, params):
    s0, k, n, a, b, c = params[0], params[1], params[2], params[3], params[4], params[5]
    return s0 + a*T + (k + b*T)*eps**n + c*np.log(eps)