def equation(t, P, params):
    return params[0]*P + params[1]*P**2 + params[2]*P*t + params[3]*t + params[4] + params[5]*t**2 + params[6]*t**2*P + params[7]*P**3 + params[8]*t*P**2