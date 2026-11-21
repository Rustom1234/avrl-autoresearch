def equation(t, x, v, params):
    return params[0]*np.cos(params[1]*t+params[2]) + params[3]/v + params[7]