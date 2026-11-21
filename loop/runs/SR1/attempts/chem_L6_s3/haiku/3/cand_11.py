def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t)*A**params[2] + params[3]*t*np.exp(-params[4]*t) - params[5]*A*t**params[6]