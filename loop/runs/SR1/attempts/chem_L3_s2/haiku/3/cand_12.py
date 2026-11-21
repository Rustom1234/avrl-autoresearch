def equation(t, A, params):
    return params[0] - params[1]*np.exp(-params[2]*t) + A**params[3]*(params[4] + params[5]*t) + params[6]*A**params[7]*np.exp(-params[8]*t)