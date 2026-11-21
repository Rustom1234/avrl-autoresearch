import numpy as np
def equation(t, P, params):
    r, K, a0, a1, a2, a3, b = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return r*P*(1-P/K) + a0 + a1*t + a2*t**2 + a3*np.exp(-b*t)
