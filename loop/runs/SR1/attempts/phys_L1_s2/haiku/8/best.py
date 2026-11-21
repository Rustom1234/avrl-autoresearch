import numpy as np
def equation(t, x, v, params):
    p=params
    return p[0]+p[1]*np.sqrt(t)+p[2]*x+p[3]*x**2+p[4]*np.sin(p[5]*x)+p[6]*np.cos(p[5]*x)+p[7]*x**3
