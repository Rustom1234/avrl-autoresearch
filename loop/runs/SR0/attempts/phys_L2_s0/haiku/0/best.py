import numpy as np
def equation(t, x, v, params):
    p=params
    E=np.exp(-p[5]*t)
    return (p[0] + p[1]*x + p[2]*v + p[3]*x**3 + p[4]*E + p[6]*x*E + p[7]*v*E + p[8]*x*x*E + p[9]/t)
