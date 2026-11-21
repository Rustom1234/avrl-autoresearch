def equation(t, x, v, params):
    return params[0] + params[1]/v**params[5] + params[2]*t**params[3]*np.exp(-params[4]*t)