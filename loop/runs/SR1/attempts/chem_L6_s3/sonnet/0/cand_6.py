def equation(t, A, params):
    return params[0]*(1-np.exp(-params[1]*t))*np.exp(-params[2]*t) - params[3]*(np.exp(params[4]*t)-1)*A**params[5]