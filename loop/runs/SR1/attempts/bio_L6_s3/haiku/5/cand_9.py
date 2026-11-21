import numpy as np
def equation(t, P, params):
    th=np.tanh(params[1]*t)
    return params[0]*th + params[2]*P + params[3]*P**2 + params[4]*P**3 + t*(params[5]+params[6]*P+params[7]*P**2) + params[8]*th*P + params[9]*th*P**2