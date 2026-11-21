def equation(t, A, params):
    return -(params[0]*A/(params[1]+A+params[2]*A**2) + params[3]*t)
