def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t) + params[2] + A**params[4]*(params[5]+params[6]*t)*np.exp(params[3]*t)