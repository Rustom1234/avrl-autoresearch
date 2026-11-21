def equation(t, x, v, params):
    return params[0]*x/(1+params[1]*x**2)**params[2] + params[3]*t**params[4] + params[5]