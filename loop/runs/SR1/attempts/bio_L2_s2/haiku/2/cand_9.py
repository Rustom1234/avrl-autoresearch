def equation(t, P, params):
    return P*(params[0]+params[1]*t)*(1-P/params[2]) * (1 - params[3]*np.exp(-params[4]*t)/P)