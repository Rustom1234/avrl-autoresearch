import numpy as np
def equation(t, x, v, params):
    w = params[9]
    return (params[0] + params[1]*x + params[2]*x**3 + params[3]*v + params[4]*v**3
            + params[5]*x*v + params[6]*x**2*v + params[7]*x*v**2
            + params[8]*0 + 0*x + (params[8]*np.cos(w*t)))  + 0*np.sin(w*t)
