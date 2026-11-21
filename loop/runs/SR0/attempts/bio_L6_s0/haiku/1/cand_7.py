def equation(t, P, params):
    return params[0]*P*t**params[1] + params[2]*P**2 + params[3]*P + params[4]*t**2 + params[5]