import numpy as np
def equation(t, P, params):
    r, K, a, b, c, d, e = params[0], params[1], params[2], params[3], params[4], params[5], params[6]
    return r*P*(1-P/K) + a/P + b*np.sqrt(P) + c*P*np.exp(-P) + d + e*t
