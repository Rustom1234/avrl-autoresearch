import numpy as np
def equation(t, x, v, params):
    p=params
    E=np.exp(-p[0]*t)
    return (p[1]*E*np.cos(p[2]*t+p[3]) + p[4] + p[5]*v + p[6]*v**2 + p[7]*v**3
            + p[8]*x + p[9]*x**3 + 0*x)