def equation(t, P, params):
    return P*(params[0] + params[1]*t + params[2]*t**2) - params[3]*P**2