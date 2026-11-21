def equation(t, A, params):
    return params[0]*(1-np.exp(-params[1]*t)) + params[2]*A**params[3]*np.exp(-params[4]*t)