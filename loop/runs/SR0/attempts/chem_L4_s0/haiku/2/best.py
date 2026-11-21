def equation(t, A, params):
    return params[0]*A*t**params[1] + params[2]/A + params[3]*A**params[4]*t**params[5] + params[6]*A**2*t**params[7]
