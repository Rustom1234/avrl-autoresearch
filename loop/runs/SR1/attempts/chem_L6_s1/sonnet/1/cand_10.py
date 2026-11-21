def equation(t, A, params):
    return params[0]*A*np.exp(-params[1]*t) + params[2]*t**params[3] + params[4]*np.exp(params[5]*t-params[6]*A)