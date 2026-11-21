def equation(t, P, params):
    r = params[0]*np.exp(-params[1]*t) + params[2]
    K = params[3]*np.exp(params[4]*t) + params[5]
    return r*P*(1 - P/K)