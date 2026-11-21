import numpy as np
def equation(t, P, params):
    s = 1/(1+np.exp(-params[0]*(t-params[1])))
    return (params[2]+params[3]*P+params[4]*P*P)*s + params[5]*P + params[6]