import numpy as np
def equation(t, x, v, params):
    w=params[0]
    return params[1] + params[2]*np.cos(w*t) + params[3]*np.sin(w*t) + params[4]*x + params[5]*x**3 + params[6]*v + params[7]*v**2 + params[8]*v**3 + params[9]*x*v
