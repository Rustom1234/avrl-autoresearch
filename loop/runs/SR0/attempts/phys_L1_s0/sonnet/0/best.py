def equation(t, x, v, params):
    return params[0]*t**params[1] + params[2]/v + params[3]*x + params[4]