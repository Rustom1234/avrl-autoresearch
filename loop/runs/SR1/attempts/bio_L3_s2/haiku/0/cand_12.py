def equation(t, P, params):
    return P*(params[0] - params[1]*P - params[2]*np.sin(params[3]*t + params[4])) + params[5] + params[6]*t + params[7]*np.sin(params[8]*t+params[9])