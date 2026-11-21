def equation(t, P, params):
    return params[0]*P+params[1]*P*P+params[2]*P**3+params[3]+params[4]*t+params[5]*t*t+params[6]*np.sin(params[7]*t+params[8])+params[9]*P*t