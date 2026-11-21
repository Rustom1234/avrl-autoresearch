import numpy as np
def equation(eps, T, params):
    p=params
    L=np.log(eps)
    return (p[0]+p[1]*eps+p[2]*eps*T+p[3]*L**2+p[4]*eps*L/T+p[5]*eps**2*T
            +p[6]*L+p[7]*L*T+p[8]*T+p[9]*T**2)
