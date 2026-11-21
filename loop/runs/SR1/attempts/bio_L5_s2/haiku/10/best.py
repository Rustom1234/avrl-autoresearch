def equation(t, P, params):
    return params[0]*np.exp(params[1]*P) + params[2]*P**3
