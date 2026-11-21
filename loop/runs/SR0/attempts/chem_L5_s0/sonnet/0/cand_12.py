def equation(t, A, params):
    return params[0]*A**params[1]*np.exp(-params[2]*A)+params[3]*A+params[4]+params[5]*np.exp(-params[6]*t)*A