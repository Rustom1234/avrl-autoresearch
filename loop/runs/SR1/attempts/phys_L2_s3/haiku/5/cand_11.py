import numpy as np
def equation(t, x, v, params):
    w = params[8]
    return (params[0] + params[1]*x + params[2]*x**2 + params[3]*x**3 + params[4]*v + params[5]*x*v
            + (params[6] + params[7]*x)*np.cos(w*t) + params[9]*np.sin(w*t)*x)
