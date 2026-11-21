def equation(t, P, params):
    return np.exp(params[0] + params[1]*t + params[2]*t**2) * P**params[3] * (1 + params[4]*P)