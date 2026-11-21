def equation(t, A, params):
    return params[0]*(1-np.exp(-params[1]*t)) + (params[2]+params[3]*np.exp(-params[4]*t))*A**params[5]+params[6]