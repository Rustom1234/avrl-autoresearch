def equation(t, x, v, params):
    return params[0]*x**params[4] + params[1]*t**params[5]/x**params[6] + params[2]*t**params[7] + params[3] + params[8]*v