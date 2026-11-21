def equation(t, A, params):
    return params[0]*np.tanh(params[1]*t)**params[2] + params[3]*A**3*np.exp(-params[4]*t)+params[5]*t