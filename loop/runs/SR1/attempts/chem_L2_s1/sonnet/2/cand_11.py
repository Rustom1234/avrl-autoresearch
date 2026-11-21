def equation(t, A, params):
    return -params[0]*A/(params[1]+A) + t**3*(params[2]*A + params[3]) + params[4]*t**3*A**2