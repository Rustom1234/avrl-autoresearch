def equation(t, P, params):
    return params[0]*P*(1+params[1]*P) + params[2]*np.sqrt(t)*P**params[3] + params[4]*t*np.exp(-params[5]*P)