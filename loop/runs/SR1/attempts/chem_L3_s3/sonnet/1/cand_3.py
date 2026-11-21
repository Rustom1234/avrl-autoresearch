def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t) + A*params[2]*np.sin(params[3]*t+params[4]) + params[5]