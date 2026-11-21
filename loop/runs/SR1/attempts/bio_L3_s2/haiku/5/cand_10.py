import numpy as np
def equation(t, P, params):
    num = params[0]+params[1]*t+params[2]*t*t
    den = 1+params[3]*t
    return (params[4]+params[5]*P+params[6]*P*P)*num/den + params[7]*P
