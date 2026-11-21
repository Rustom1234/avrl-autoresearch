def equation(t, P, params):
    return params[0]*P*np.sqrt(t) + params[1]*np.sqrt(P) + params[2]*t/(1+params[3]*t)