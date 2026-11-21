def equation(t,P,params):
    return params[0]*P+params[1]/P+params[2]*P**params[3]*np.exp(-params[4]*t)+params[5]