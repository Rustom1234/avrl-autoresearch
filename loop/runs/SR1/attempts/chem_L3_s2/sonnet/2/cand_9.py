def equation(t, A, params):
    return params[0]*(1-np.exp(-params[1]*t)) + params[6]*t + params[2]*A**params[4]*np.exp(-params[3]*t)