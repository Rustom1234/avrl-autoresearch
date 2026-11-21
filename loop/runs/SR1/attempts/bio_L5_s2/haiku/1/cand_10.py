def equation(t, P, params):
    return (params[0] + params[1]*P + params[2]*P**2)/(1 + params[3]*P + params[4]*P**2)
