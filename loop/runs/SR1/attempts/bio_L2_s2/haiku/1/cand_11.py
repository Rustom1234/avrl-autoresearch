def equation(t, P, params):
    return P*(params[0] + params[1]*P + params[2]*t + params[3]*t*P + params[4]*P**2*t)/(1 + params[5]*t)