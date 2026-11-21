def equation(t, P, params):
    base = params[0] + params[1]*P + params[2]*P**2 + params[3]*P**3
    e = np.exp(params[4]*t)
    return base + e*(params[5] + params[6]*P + params[7]*P**2)