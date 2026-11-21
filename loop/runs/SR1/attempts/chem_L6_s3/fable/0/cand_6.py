def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t)/(1+np.exp(-params[2]*(t-params[7]))) - params[3]*A**params[4]*t**params[5] + params[6]