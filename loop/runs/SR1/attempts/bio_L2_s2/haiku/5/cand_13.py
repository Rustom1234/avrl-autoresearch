def equation(t, P, params):
    s = np.sin(params[3]*t + params[4])
    return P*(params[0] - params[1]*P) + params[2]*s + params[5]*P*s