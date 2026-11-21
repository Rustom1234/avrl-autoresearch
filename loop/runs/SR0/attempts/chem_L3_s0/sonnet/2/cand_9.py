def equation(t, A, params):
    return params[0] + params[1]*A + params[2]*A*np.log(t+params[3])+params[4]*np.exp(params[5]*A)