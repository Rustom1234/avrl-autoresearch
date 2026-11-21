def equation(t,P,params):
    return params[0]*P**params[1]+params[2]+params[3]*t+params[4]*np.exp(-params[5]*t)+params[6]*P