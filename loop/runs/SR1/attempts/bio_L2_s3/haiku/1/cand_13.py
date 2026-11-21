def equation(t, P, params):
    base = params[0] + params[1]*P + params[2]*P**2 + params[3]*P**3
    a = params[4] + params[5]*t + params[6]*t**2
    return base + a*(P + params[7]*P**2 + params[8]*P**3)