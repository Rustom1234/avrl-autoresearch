import numpy as np
def equation(t, x, v, params):
    p=params
    w=0.4
    s=np.sin(w*t); c=np.cos(w*t); s2=np.sin(2*w*t); c2=np.cos(2*w*t)
    return p[0]+p[1]*x+p[2]*x**3+p[3]*v+p[4]*s+p[5]*c+p[6]*s*v+p[7]*c*v+p[8]*s2+p[9]*c2
