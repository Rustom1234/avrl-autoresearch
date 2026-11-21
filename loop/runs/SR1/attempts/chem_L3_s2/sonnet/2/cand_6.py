def equation(t, A, params):
    return params[0]*t/(params[1]+t) + params[2]*A**3/(1+params[3]*t) + params[4]*A**params[5]