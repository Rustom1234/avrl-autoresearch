def equation(t, P, params):
    return -P*(params[0] + params[1]*P**params[2] + params[3]*t**2*P**params[4])
