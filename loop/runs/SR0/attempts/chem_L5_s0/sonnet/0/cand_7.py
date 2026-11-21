def equation(t, A, params):
    return params[0]*A/(1+params[1]*A**2)+params[2]*A+params[3]