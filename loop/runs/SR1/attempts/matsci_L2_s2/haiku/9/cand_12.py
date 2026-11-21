import numpy as np
def equation(eps, T, params):
    p=params
    L=np.log(eps); s=np.sqrt(eps)
    return (p[0]+p[1]*T+p[2]*T**2) + (p[3]+p[4]*T)*L + (p[5]+p[6]*T)*eps + (p[7]+p[8]*T)*s + p[9]*eps**2
