def equation(t, P, params):
    return params[0]*np.sqrt(P) + params[1]*t*np.exp(-params[2]*t) + params[3]