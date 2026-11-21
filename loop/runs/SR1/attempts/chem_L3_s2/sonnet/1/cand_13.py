def equation(t, A, params):
    return params[0]*np.sin(params[1]*t**params[5])+params[4]+params[2]*A**3*np.exp(-params[3]*t)