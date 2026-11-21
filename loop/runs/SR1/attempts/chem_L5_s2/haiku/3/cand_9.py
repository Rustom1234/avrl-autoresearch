def equation(t, A, params):
    return params[0] + params[1]*t + params[2]*t**params[3] + params[4]*A + params[5]*A*t + params[6]*A**params[7]
