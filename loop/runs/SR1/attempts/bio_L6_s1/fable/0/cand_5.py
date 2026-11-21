def equation(t, P, params):
    return params[0]*np.exp(params[1]*np.sqrt(P)) + params[2]*P**2 + params[3]*np.sqrt(P)*t**2