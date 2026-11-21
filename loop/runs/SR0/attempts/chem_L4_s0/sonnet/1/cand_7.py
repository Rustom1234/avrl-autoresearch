def equation(t, A, params):
    return params[0]*(1-np.exp(-params[1]*t))*A/(params[2]+A) - params[3]*A*t - params[4]*np.exp(-params[5]*A)+params[6]