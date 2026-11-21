def equation(t, P, params):
    return P*(params[0] + params[1]*t + params[2]*P + params[3]*t*P + params[4]*P**2 + params[5]*t**2)