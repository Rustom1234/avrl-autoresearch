def equation(t, P, params):
    return -P**params[0]*(params[1] + params[2]*t*t)
