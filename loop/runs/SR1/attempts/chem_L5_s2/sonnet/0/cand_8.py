def equation(t, A, params):
    return params[0]+params[1]*np.sqrt(t)+params[2]*np.exp(-params[3]*t)