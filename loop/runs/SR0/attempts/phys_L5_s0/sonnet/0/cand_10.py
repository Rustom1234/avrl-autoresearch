def equation(t, x, v, params):
    return params[0] + params[1]/v + params[2]*np.exp(-params[3]*t) + params[4]*np.cos(params[5]*t+params[6]) + params[7]*x