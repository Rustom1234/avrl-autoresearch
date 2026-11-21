def equation(t, P, params):
    return params[0]*np.tanh(params[1]*t) + params[2]*P**2 + params[3]*P**3*np.abs(t)**params[4]