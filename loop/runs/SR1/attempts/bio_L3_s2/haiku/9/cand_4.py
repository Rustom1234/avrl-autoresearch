def equation(t, P, params):
    r = params[0] + params[1]*t + params[2]*t**2
    return r*P*(1 - P/params[3])