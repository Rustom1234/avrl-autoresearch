def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t) - params[5]*np.exp(-params[6]*t) - params[2]*t**params[3]*A**params[4]