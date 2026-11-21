def equation(t, A, params):
    return params[0] + params[1]*t + params[2]*t**2 - params[3]*A**params[4] + params[5]*t*A
