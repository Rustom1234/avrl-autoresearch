def equation(t, A, params):
    return params[0]*(1-np.exp(-params[1]*t)) + params[2]*np.sin(params[3]*t+params[5]) + params[4]*A**3*np.exp(-params[6]*t)