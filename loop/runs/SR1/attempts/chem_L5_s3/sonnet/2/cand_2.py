def equation(t, A, params):
    return params[0]*A**params[1]*np.exp(-params[2]*t)*t**params[3]+params[4]*np.sin(params[5]*t)+params[6]