def equation(t, P, params):
    return P*(params[0]+params[1]*np.sin(params[2]*t+params[3]))-params[4]*P*P*(1+params[5]*t)+params[6]*np.sin(params[7]*t+params[8])+params[9]