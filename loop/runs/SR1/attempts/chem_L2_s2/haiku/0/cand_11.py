import numpy as np
def equation(t, A, params):
    Ap = np.abs(A)**params[2]
    K = params[3]*(1+params[4]*t)
    return (A - params[0])*params[1]*Ap/(K + Ap)*np.exp(-params[5]*t)
