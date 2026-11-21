def equation(t, P, params):
    r = params[0] + params[1]*t + params[2]*t*t
    return r*P*(1-P/params[3]) + params[4]/P + params[5] + params[6]*P
