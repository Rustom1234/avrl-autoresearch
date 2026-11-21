def equation(t, A, params):
    base = params[0]*(1-params[1]*np.exp(-params[2]*t))
    return base + params[3]*A**params[4]*(1-params[5]*np.exp(-params[6]*t))