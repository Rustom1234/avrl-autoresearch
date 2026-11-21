def equation(t, A, params):
    return params[0]+params[1]*t+params[2]*np.exp(-params[3]*t)+params[4]*np.exp(-params[5]*t)