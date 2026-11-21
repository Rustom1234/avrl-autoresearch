def equation(t, A, params):
    return params[0]*(1-np.exp(-params[1]*t))*np.exp(-params[5]*t)+params[4]+params[2]*A**3*np.exp(-params[3]*t)