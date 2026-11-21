def equation(t, A, params):
    return params[0]*(1-np.exp(-params[1]*t))+params[4]+params[2]*A**params[5]*np.exp(-params[3]*t)