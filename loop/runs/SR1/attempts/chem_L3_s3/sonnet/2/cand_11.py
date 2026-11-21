def equation(t, A, params):
    return params[0]*A**2*t**params[1] + params[2] + params[3]/t + params[4]*np.exp(-params[5]*t)+params[6]*t