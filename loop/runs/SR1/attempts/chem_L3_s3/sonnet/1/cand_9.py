def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t) + params[2] + params[3]*t + A**params[4]*(params[5]+params[6]*t)