def equation(t, A, params):
    return params[0]*A + params[1]*A**2 + params[2]*t**2 + params[3]*t**2*A + params[4]*t*A + params[5]*t*A**2 + params[6]*t
