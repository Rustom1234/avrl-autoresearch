def equation(t, x, v, params):
    return params[0]*t*np.exp(-params[1]*t) + params[2]/v + params[3]