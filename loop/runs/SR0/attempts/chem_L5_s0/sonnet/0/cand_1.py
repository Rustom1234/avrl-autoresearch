def equation(t, A, params):
    return params[0]*np.sin(params[1]*A+params[2])+params[3]*np.exp(-params[4]*t)*A+params[5]