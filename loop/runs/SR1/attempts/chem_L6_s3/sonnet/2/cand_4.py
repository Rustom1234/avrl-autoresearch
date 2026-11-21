def equation(t, A, params):
    return params[0]*t*np.exp(-params[1]*t) - params[2]*t**params[3]*A**params[4]