def equation(t, P, params):
    return -(params[0]*P**params[1] + params[2]*t*t*P**params[3] + params[4]*t*P**params[5])
