def equation(t, A, params):
    return params[0]*A**params[1]*np.exp(-params[2]*A*t) + params[3]*A**params[4]*t**params[5]
