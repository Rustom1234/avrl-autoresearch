def equation(t, A, params):
    return params[0] + params[1]*t + params[2]*t**2 + (params[3] + params[4]*t + params[5]*t**2)*A**2 + params[6]*A**3 + params[7]*t*A**3 + params[8]*A + params[9]*t*A
