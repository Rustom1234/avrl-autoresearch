def equation(t, P, params):
    return params[0] + params[1]*P + params[2]*t*np.exp(-params[3]*t) + params[4]*(1-np.exp(-params[5]*t))
