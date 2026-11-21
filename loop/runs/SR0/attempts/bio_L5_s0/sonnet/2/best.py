def equation(t, P, params):
    return params[0]*P**params[1] - params[2]*t*np.exp(-params[3]*t) + params[4]