import numpy as np
def equation(t, x, v, params):
    p=params
    w=p[9]
    c=np.cos(w*t); s=np.sin(w*t)
    return p[0]+p[1]*x+p[2]*v+p[3]*x**3+(p[4]+p[5]*v+p[6]*t)*c+(p[7]+p[8]*v+p[10]*t)*s
