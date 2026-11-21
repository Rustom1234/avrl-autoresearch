import numpy as np
def equation(t, P, params):
    r, K0, a, w, phi, b = params[0], params[1], params[2], params[3], params[4], params[5]
    K = K0*(1 + a*np.sin(w*t+phi))
    return r*P*(1-P/K) - b*P
