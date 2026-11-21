def equation(t, A, params):
    return params[0] + params[1]*A + params[2]*t*A + params[3]*t**2*A + params[4]*A**2 + params[5]*t + params[6]*t**2 + params[7]*A*t**3 + params[8]*A**2*t