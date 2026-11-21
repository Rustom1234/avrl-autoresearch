def equation(t, P, params):
    g = 1 + params[3]*t + params[4]*t**2
    return -(params[0]*P + params[1]*P**2 + params[2]*P**1.5 + params[5]*P**2.5)*g - params[6]*P**1.5*t - params[7]*P**2*t