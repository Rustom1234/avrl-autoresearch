def equation(t, A, params):
    return params[0]*t/(params[1]+t) + params[2]*A**params[3]
