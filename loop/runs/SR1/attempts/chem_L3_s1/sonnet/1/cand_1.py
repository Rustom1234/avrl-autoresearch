def equation(t, A, params):
    return params[0]*np.exp(-params[1]*A**params[2]) *np.cos(params[3]*t)+params[4]*A