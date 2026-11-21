import numpy as np
def equation(eps, T, params):
    p=params
    S=1-np.exp(-p[3]*eps)
    L=np.log(eps)
    return p[0]+p[1]*L+p[2]*S+(p[4]+p[5]*T+p[6]*T**2)*L*S+(p[7]+p[8]*T+p[9]*T**2)*S*T
