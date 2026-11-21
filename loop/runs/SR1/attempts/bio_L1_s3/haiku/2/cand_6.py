def equation(t, P, params):
    return params[0] + params[1]*P + params[2]*P**2 + params[3]*P/(1+P)
