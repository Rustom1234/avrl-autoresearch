def equation(t, A, params):
    return params[0] + params[1]*A**params[2] + params[3]*t**params[4]*A**params[5] + params[6]*A**params[7]*t**params[8]