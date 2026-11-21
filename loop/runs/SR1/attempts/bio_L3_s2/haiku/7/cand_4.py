def equation(t, P, params):
    return params[0] + params[1]*t + P*(params[2] + params[3]*t) + params[4]*P**2 + params[5]*P**2*t
