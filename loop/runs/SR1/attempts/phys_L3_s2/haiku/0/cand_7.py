import numpy as np
def equation(t, x, v, params):
    w = params[9]
    return (params[0] + params[1]*x + params[2]*v + params[3]*x**2 + params[4]*x**3
            + params[5]*x*v + params[6]*x*x*v + params[7]*v*np.abs(v)
            + params[8]*np.cos(w*t) + params[8]*0 )
