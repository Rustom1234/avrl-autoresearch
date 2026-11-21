def equation(t, P, params):
    return params[0]*P + params[1]*P**2 + params[2]*t**params[3] + params[4]*P**params[5]*t**params[6]