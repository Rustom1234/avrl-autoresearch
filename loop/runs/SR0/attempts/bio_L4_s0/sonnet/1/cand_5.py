def equation(t,P,params):
    return params[0]*P**params[1]*(1+params[2]*np.sin(params[3]*t+params[4]))+params[5]*t+params[6]*P*np.exp(-params[7]*t)