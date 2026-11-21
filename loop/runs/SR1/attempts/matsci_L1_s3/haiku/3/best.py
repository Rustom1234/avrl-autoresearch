import numpy as np
def equation(eps, T, params):
    return -(params[0]*T**params[1] + params[2]*T**params[3]*eps**(params[4]+params[5]*T) + params[6]*T**params[7]*eps*0 + params[8]*eps**params[9]*0)
