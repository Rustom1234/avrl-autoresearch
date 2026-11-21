def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t**2) - params[2]*A**params[3]
