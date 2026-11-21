import numpy as np
def equation(t, P, params):
    g1 = t/(1+t)
    g2 = np.exp(-0.3*t)
    g3 = np.exp(-0.6*t)
    g4 = t
    g5 = np.sqrt(t)
    g6 = 1/(1+t)**2
    g7 = 1/(1+t)
    return (params[0]*g1 + params[1]*P**3*g2 + params[2]*g3 + params[3]*P**3*g4
            + params[4]*P*g5 + params[5]*g6 + params[6]*P**2*g7)
