def equation(t, A, params):
    return params[0]*np.sqrt(A) - params[1]*A*np.exp(-params[2]*A)