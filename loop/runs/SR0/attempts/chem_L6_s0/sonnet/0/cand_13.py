def equation(t, A, params):
    return params[0]*A**2*(1+params[1]/t**params[2]) + params[3]*t**params[4]+params[5]*t*A