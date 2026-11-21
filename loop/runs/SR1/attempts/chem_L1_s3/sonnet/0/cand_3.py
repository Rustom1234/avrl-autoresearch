def equation(t, A, params):
    return -(params[0]/A + params[1]*A**params[2] + params[3]*A + params[4]*np.exp(-params[5]*t))