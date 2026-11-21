def equation(t, A, params):
    return params[0]*(1-np.exp(-params[1]*t)) + params[2]*np.exp(-params[3]*t)*A**3+params[4]*t+params[5]