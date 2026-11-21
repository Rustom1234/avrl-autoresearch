def equation(t, P, params):
    return params[0]+params[1]*t+params[2]*P+params[3]*P*np.sin(params[4]*t+params[5])+params[6]*P*P+params[7]*np.sin(params[4]*t+params[8])