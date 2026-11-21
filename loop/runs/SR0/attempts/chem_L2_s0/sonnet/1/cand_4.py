def equation(t, A, params):
    return -params[0]*A**(-params[1]) + params[2]*A**params[4]*np.exp(-params[3]*t)