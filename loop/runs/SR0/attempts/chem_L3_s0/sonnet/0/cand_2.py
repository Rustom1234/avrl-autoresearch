def equation(t, A, params):
    return params[0]+params[1]*A*np.exp(-params[2]*t)+params[3]*A**2*np.exp(-params[4]*t)+params[5]*A*t