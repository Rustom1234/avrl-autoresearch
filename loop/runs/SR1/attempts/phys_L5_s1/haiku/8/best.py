def equation(t, x, v, params):
    return params[0]*np.exp(-params[1]*t) - params[2]*np.sqrt(v) + params[3]*x