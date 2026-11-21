def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t)*(1-np.exp(-params[2]*t**2)) - params[3]*A*t**params[4]*np.exp(-params[5]*t)+params[6]*A**2*t