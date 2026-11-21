import numpy as np
def equation(t, A, params):
    Ap = np.abs(A)**params[2]
    return (A - params[0])*params[1]*Ap/(Ap + params[3]*t)*np.exp(-params[4]*t)
