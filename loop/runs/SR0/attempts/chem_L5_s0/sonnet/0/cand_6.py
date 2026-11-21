def equation(t, A, params):
    return params[0]*A*np.exp(-params[1]*A)+params[2]*A+params[3]