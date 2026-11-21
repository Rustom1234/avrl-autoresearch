def equation(eps, T, params):
    return params[0]*eps**params[1] + params[2]*T**params[3] + params[4] + params[5]*eps**params[6]*T**params[7]