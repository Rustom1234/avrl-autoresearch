import numpy as np
def equation(eps, T, params):
    return -(params[0]*eps+params[1]*eps**2)*(1+params[2]*T+params[3]*T**2)/(1+params[4]*eps*T)
