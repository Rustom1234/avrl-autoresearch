def equation(t, P, params):
    return params[0]*P**params[1] + params[2]*t + params[3]*P*np.exp(-params[4]*t) + params[5]