def equation(t, P, params):
    return params[0] + params[1]*P + params[2]*P*np.exp(-params[3]*P)
