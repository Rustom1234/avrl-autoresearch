def equation(t, x, v, params):
    return params[0]*np.sin(params[1]*t)+params[2]*t+params[3]