def equation(t, A, params):
    f = params[0]*A/(params[1]+A) + params[2]
    return f - t**2*(params[3] + params[4]/A**params[5]) + params[6]*t