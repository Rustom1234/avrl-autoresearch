def equation(t, P, params):
    r = params[0] + params[1]*t + params[2]*t**2
    return r*P - params[3]*P**2 + params[4]*t*P**2
