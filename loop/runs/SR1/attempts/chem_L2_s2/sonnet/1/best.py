def equation(t, A, params):
    return params[0] + params[1]*A**params[2] + params[3]*t**params[4]*A**(-params[7]) + params[5]*np.exp(-params[6]*A)+params[8]*A*np.exp(-A)