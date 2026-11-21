def equation(t, P, params):
    r = params[0] + params[1]*t
    K = params[2] + params[3]*t
    return r*P*(1 - P/K)