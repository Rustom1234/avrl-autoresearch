def equation(t, A, params):
    return params[0]*np.sqrt(A)+params[1]*A+params[2]*A**2+params[3]*np.exp(-params[4]*t)