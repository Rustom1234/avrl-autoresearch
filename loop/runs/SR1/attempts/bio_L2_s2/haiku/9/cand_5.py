def equation(t, P, params):
    return P*(params[0] - params[1]*P)*(1 + params[2]*np.sin(params[3]*t + params[4])) + params[5]*np.exp(-params[6]*t)*P