def equation(eps, T, params):
    p=params[7]
    return params[0] + params[1]*T + params[2]*eps**p + params[3]*eps**p/T + params[4]*eps/T + params[5]*eps + params[6]/T
