import numpy as np
def equation(t, P, params):
    return P*(params[0] + params[1]*t + params[2]*P + params[3]*t*P + params[4]*t*t + params[5]*t*t*P)