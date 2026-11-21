def equation(t, P, params):
    return -(params[0]*P**params[1]*(1+params[2]*t*t) + params[3]*P**params[4])
