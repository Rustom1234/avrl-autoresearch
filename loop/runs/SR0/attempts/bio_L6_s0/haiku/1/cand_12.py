def equation(t, P, params):
    return P*(params[0]*np.exp(params[1]*t)+params[2]) + params[3]*P**2 + params[4]*P**3 + params[5]*np.exp(params[1]*t)*P**2