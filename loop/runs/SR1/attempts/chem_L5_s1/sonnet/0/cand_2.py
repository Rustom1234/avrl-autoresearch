def equation(t, A, params):
    return -params[0]*t**params[1]*A**params[2]*np.exp(-params[3]*A) - params[4]*A**params[5]