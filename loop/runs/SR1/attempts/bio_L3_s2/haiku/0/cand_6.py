def equation(t, P, params):
    return P*(params[0] + params[1]*np.sin(params[2]*t + params[3]) - params[4]*P)