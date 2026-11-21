def equation(t, A, params):
    return params[0]*A**params[1] + params[2] + params[3]*np.exp(-params[4]*t)/t**params[5]