def equation(t, A, params):
    return params[0]*np.tanh(params[1]*t)+params[2]*A**params[3]+params[4]