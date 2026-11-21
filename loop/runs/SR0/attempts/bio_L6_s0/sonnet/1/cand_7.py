def equation(t, P, params):
    return params[0]*P + params[1]*P**2*t + params[2]*t + params[3]*P*t + params[4]*P**2 + params[5]