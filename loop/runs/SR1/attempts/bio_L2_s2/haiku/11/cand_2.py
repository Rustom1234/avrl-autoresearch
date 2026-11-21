def equation(t, P, params):
    r = params[0] + params[1]*np.sin(params[2]*t + params[3])
    return r*P*(1 - P/params[4]) - params[5]*P**2 + params[6]*t*P
