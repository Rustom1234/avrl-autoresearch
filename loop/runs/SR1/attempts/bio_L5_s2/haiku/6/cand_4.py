def equation(t, P, params):
    return params[0]*np.exp(-params[1]*P) + params[2] + params[3]*P + params[4]*P**2 + params[5]*P**3