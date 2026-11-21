import numpy as np
def equation(t, P, params):
    L = np.log(P)
    p = params
    return (p[0] + p[1]*P + p[2]*P*P + p[3]*P*L + p[4]*L
            + t*(p[5] + p[6]*P + p[7]*P*L) + p[8]*t*t + p[9]*t*t*P)
