def equation(t, A, params):
    return params[0]*np.sin(params[1]*t+params[2])+params[3]+params[4]*A**2*np.exp(-params[5]*t)+params[6]*A