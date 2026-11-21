import numpy as np
def equation(t, x, v, params):
    w=params[0]
    return (params[1]+params[2]*x+params[3]*v+params[4]*x**2+params[5]*v*np.abs(v)+params[6]*x**3+params[7]*v**3+params[8]*x*v
            +params[9]*np.cos(w*t)+ 0.0*np.sin(w*t))
