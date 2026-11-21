def equation(t, A, params):
    return params[0]*(1-np.exp(-params[1]*t)) + params[2]*A + params[3]*A**2 + params[4]*np.sqrt(t)*A