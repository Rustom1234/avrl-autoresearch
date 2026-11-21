def equation(t, A, params):
    return params[0] + params[1]*(1-np.exp(-params[2]*t)) + params[3]*A**params[4]*t*np.exp(-params[5]*t)