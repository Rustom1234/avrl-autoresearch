import numpy as np
def equation(t, x, v, params):
    p=params
    w=p[9]
    return (p[0]+p[1]*x+p[2]*v+p[3]*x**2+p[4]*x**3
            +(p[5]+p[6]*v)*np.cos(w*t)+(p[7]+p[8]*v)*np.sin(w*t))
