def equation(t, A, params):
    return params[0]*A**params[1]/(1+params[2]*t)