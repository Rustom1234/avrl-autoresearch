def equation(t, A, params):
    return params[0]*A**params[1]/(1+params[2]*A**params[6])**2+params[3]*t**params[4]+params[5]