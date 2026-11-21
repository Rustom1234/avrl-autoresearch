def equation(t, A, params):
    return params[0] + params[1]*A**params[2] + params[3]*A**params[4]*t**params[5] + params[6]*A**params[7]*np.exp(-params[8]*t)