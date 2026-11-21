def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t**params[2])*np.exp(-params[5]*A**params[6]) - params[3]*A**params[4]