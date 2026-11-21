import numpy as np
def equation(eps, T, params):
    return (params[0]+params[1]*eps**params[2])*(params[3]+params[4]*T**params[5])+params[6]*T+params[7]
