def equation(t, P, params):
    return P*(params[0]+params[1]*P+params[2]*P**2+params[3]*t+params[4]*t*P+params[5]*t*P**2+params[6]*t**2*P)