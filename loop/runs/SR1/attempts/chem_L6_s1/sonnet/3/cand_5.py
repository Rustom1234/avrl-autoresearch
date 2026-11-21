def equation(t, A, params):
    return params[0]*A/(params[1]+A) + params[2]*t**2 + params[3]*t**params[4]*A**(-params[5])