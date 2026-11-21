def equation(t, A, params):
    f = params[0]*A**params[7]/(params[1]+A**params[7]) + params[2]
    return f - t**params[8]*(params[3] + params[4]/A**params[5]) + params[6]*t