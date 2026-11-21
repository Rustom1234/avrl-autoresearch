def equation(t, A, params):
    return -params[0]*t**params[1]*A/(1+params[2]*A*t)
