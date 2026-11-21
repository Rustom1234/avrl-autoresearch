def equation(t, A, params):
    return params[0]*t**params[1]*np.exp(-params[2]*t)*A**params[3] + params[4]*t*np.exp(-params[5]*t) - params[6]*A**params[7]*t**params[8] + params[9]