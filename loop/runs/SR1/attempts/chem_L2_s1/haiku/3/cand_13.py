def equation(t, A, params):
    return params[0]*A**2 + params[1]*A + params[2] + t**params[3]*(params[4] + params[5]*A + params[6]*A**2) + params[7]*t**params[8]*A
