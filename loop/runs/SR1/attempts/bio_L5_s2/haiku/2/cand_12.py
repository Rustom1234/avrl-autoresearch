def equation(t, P, params):
    return params[0]*(params[1]-P)*np.exp(-params[2]*P)+params[3]*P*(params[1]-P)
