def equation(t, A, params):
    return params[0]*t**(-params[1]) + params[2] + params[3]*A**2*t**params[5]+params[4]*A*t