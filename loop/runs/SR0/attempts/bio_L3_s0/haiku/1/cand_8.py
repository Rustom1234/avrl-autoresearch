def equation(t, P, params):
    return params[0] + params[1]*t*np.exp(-params[2]*P) + params[3]*P + params[4]*P*P + params[5]*t