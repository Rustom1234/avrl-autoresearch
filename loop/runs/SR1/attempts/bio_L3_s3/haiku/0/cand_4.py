def equation(t, P, params):
    F = params[0] + params[1]*P + params[2]*P**2
    G = (1 - np.exp(-params[3]*t)) * (1 + params[4]*np.sin(params[5]*t + params[6]))
    return F*G
