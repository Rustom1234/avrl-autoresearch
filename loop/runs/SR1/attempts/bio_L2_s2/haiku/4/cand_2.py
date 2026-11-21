def equation(t, P, params):
    return P*(params[0]*np.exp(-params[1]*t) - params[2]*P*np.exp(-params[3]*t)) + params[4]*np.exp(-params[5]*t)