import numpy as np
def equation(t, x, v, params):
    s=np.sqrt(t)
    return (params[0]+params[1]*x+params[2]*x**2+params[3]*x**3+params[4]*v+params[5]*v*np.abs(v)
            +params[6]*s+params[7]*t+params[8]*t*x+params[9]*s*x)
