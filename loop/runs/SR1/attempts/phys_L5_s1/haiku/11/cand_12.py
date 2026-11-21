def equation(t, x, v, params):
    return params[0] + params[1]*x + params[2]*v + params[3]*v*np.abs(v) + params[4]*np.exp(-params[5]*t) + params[6]/v + params[7]*v**3