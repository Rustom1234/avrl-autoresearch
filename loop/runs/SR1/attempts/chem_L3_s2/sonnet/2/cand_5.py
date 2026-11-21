def equation(t, A, params):
    return params[0]*(1-np.exp(-params[1]*t)) + params[2]*A**3*np.exp(-params[3]*t) + params[4]*A**params[5]+params[6]