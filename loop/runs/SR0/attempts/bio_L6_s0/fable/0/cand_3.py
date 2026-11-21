def equation(t, P, params):
    return params[0]*P+params[1]*P**2+params[2]*P**3+params[3]*t**params[4]+params[5]+params[6]*P**params[7]*t**params[8]