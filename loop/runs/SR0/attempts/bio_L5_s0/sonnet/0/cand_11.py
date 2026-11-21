def equation(t,P,params):
    return params[0]*P**params[1]+params[2]+params[3]*np.sin(params[7]*t+params[8])+params[4]*np.exp(-params[5]*t)+params[6]*t