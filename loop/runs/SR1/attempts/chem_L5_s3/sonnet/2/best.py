def equation(t, A, params):
    return params[0]*np.sqrt(A)+params[1]*t*np.exp(-params[2]*t)