def equation(t, P, params):
    s = 1 - np.exp(-params[4]*t)
    return params[0] + params[1]*P + params[2]*P**2 + s*(params[3] + params[5]*P + params[6]*P**2) + params[7]*t
