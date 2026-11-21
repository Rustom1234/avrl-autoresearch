def equation(t, A, params):
    return params[0]*A**params[1]*t**params[2]/(1+params[3]*t**params[4])