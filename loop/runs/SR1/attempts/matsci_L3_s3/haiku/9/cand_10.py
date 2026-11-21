import numpy as np
def equation(eps, T, params):
    return (params[0] + params[1]*T)*(1+params[2]*eps)**(params[3]+params[4]*T) + params[5]*T**params[6]
