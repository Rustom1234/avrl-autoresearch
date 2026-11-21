def equation(t, A, params):
    return params[0]/t + params[1] + params[2]*t + A**params[3]*(params[4]*(1-np.exp(-params[5]*t))+params[6]*t)