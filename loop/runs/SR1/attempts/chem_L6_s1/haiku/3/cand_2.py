def equation(t, A, params):
    a = params[0] + params[1]*t + params[2]*t**2
    b = params[3] + params[4]*t
    return a*A**params[5] + b*A**2