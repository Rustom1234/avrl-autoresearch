def equation(t, A, params):
    return params[0]*(1-np.exp(-params[1]*t)) + params[6]*t + (params[2]+params[3]*t)*A**params[4]