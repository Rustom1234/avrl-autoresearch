def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t) - params[2]*np.exp(-params[4]*t)*A**params[3]
