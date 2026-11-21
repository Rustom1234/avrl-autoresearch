import numpy as np
def equation(t, x, v, params):
    p=params
    return (p[0]+p[1]*x+p[2]*x**2+p[3]*x**3+p[4]*v+p[5]*v**2+p[6]*np.log(v)
            +p[7]*x*v+p[8]*x*np.log(v)+p[9]*np.cos(3.38*t))
