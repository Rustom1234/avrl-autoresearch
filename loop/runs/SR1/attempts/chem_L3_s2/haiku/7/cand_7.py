def equation(t, A, params):
    return params[0]*t/(params[1]+t)*(1+params[2]*A**params[3])
