def equation(t, P, params):
    return P*(params[0] - params[1]*P) + params[2]*np.cos(params[3]*t + params[4])*P + params[5]*np.cos(params[6]*t+params[7])
