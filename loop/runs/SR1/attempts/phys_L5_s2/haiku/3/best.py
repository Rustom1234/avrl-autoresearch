import numpy as np
def equation(t, x, v, params):
    A, w, k = params[0], params[1], params[2]
    return A*np.sin(w*t) - k*t
