import numpy as np
def equation(t, P, params):
    r, K, c, b, e0, e1, e2, w, phi = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]
    return r*P*(1-P/K) - c*P**2/(1+b*P) + e0 + e1*t + e2*np.sin(w*t+phi)
