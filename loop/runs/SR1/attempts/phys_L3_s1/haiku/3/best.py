import numpy as np
def equation(t, x, v, params):
    w=params[0]
    c=np.cos(w*t); s=np.sin(w*t)
    return params[1] + params[2]*v + params[3]*v**2 + (params[4]+params[5]*v)*c + (params[6]+params[7]*v)*s + params[8]*x + params[9]*x**2
