import numpy as np
def equation(t, P, params):
    k=0.29
    th=np.tanh(k*t); e=np.exp(-k*t)
    return (params[0]*th + params[1]*P*e + params[2]*t*th*P**3 + params[3]*t*P**2 + params[4]*e
            + params[5]*t**2*P**3 + params[6]*th*P**2 + params[7]*P**2 + params[8]*P**3 + params[9]*t*e)