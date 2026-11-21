import numpy as np
def equation(eps, T, params):
    L=np.log(eps); s=np.sqrt(eps)
    return (params[0] + params[1]*T + params[2]*T**2 + params[3]*T**3
            + params[4]*s + params[5]*s*T + params[6]*s*T**2
            + params[7]*L + params[8]*L*T + params[9]*L*T**2)
