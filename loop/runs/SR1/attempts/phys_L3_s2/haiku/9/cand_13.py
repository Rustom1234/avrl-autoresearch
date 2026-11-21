import numpy as np
def equation(t, x, v, params):
    w=params[10]
    return (params[0] + params[1]*x + params[2]*v + params[3]*x**2 + params[4]*x*v + params[5]*v**2
            + params[6]*x**3 + params[7]*x**2*v + params[8]*x*v**2 + params[9]*v**3
            + params[11]*np.cos(w*t) + params[12]*np.sin(w*t))