import numpy as np
def equation(eps, T, params):
    p=params
    L=np.log(eps)
    return (p[0]+p[1]*T+p[2]*T**2)*L+(p[3]+p[4]*T)*eps+(p[5]+p[6]*T+p[7]*T**2)*eps**2+(p[8]+p[9]*T)/eps
