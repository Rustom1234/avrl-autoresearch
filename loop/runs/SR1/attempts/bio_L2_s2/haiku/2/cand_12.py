def equation(t, P, params):
    return params[0]*np.exp(-params[1]*t)*P**params[2] + params[3]*P - params[4]*P**2