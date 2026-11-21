def equation(t, A, params):
    return -(params[0] + params[1]*A**params[2]*t**params[3] + params[4]*A**params[5]*np.exp(-params[6]*t))