import numpy as np
def equation(t, P, params):
    return params[0]/(1+(P/params[1])**params[2]) + params[3]*t*P/(params[4]+P) + params[5]*t + params[6]*t*t
