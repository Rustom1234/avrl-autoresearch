import numpy as np
def equation(t, P, params):
    s = t**params[0]
    return (params[1]+params[2]*P+params[3]*P*P)*s + params[4] + params[5]*P + params[6]*P*P*s*0
