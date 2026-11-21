def equation(t, A, params):
    return params[0]+params[1]*np.log(t+params[4])+params[2]*A**2*np.exp(-params[3]*t)+params[5]*A