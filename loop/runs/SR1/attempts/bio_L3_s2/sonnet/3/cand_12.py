def equation(t, P, params):
    return params[0]+params[1]*t+params[2]*np.sqrt(P)+params[3]*P*np.sin(params[4]*t)+params[5]*np.sin(params[6]*t+params[7])