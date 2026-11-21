import numpy as np
def equation(t, x, v, params):
    p=params
    w=p[9]
    return (p[0]+p[1]*x+p[2]*v+p[3]*x**2+p[4]*x**3
            +(p[5]+p[6]*v+p[8]*x)*np.cos(w*t)+(p[7]+p[10]*v+p[11]*x)*np.sin(w*t))
