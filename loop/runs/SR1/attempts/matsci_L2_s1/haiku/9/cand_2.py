import numpy as np
def equation(eps, T, params):
    return -(params[0] + params[1]*T**params[2]) * (eps**params[3]) ** 0 * (1 + params[4]*eps**params[5]) 
