def equation(t, P, params):
    return params[0] + params[1]*P/(params[2]+P)*(1+params[3]*np.cos(params[4]*t+params[5])) + params[6]*t