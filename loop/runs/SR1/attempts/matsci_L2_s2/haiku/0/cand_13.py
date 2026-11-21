import numpy as np
def equation(eps, T, params):
    p=params
    L=np.log(eps)
    return (p[0]+p[1]*T)*L+(p[2]+p[3]*T)*eps+p[4]*eps**2+(p[5]+p[6]*T)/eps+p[7]+p[8]*T+p[9]*eps*L
