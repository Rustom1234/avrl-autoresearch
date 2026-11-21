import numpy as np
def equation(t, x, v, params):
    w=params[0]
    C=np.cos(w*t); S=np.sin(w*t)
    return (params[1]+params[2]*x+params[3]*v+params[4]*x**2+params[5]*v*np.abs(v)+params[6]*x**3+params[7]*v**3
            +params[8]*C+params[9]*S+ 0.0*x*v)
