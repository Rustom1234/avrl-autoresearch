def equation(t, A, params):
    return -(params[0]*A/(params[1]+A**2)+params[2]*t**params[3])