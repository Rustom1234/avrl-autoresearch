def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t)+params[2]*t+params[3]+params[4]*A*np.exp(-params[5]*t)