def equation(t, P, params):
    return params[0]*P + params[1]*t*P + params[2]*P**2 + params[3]*t*P**2 + params[4]*t + params[5]
