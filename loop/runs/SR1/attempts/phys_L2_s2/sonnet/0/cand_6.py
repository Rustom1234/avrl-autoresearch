def equation(t, x, v, params):
    return params[0] + params[1]*x + params[2]*x**(-params[3]) + params[4]*v**(-params[5]) + params[6]*np.cos(params[7]*t)+params[8]*np.sin(params[7]*t)