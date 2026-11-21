def equation(t,P,params):
    return params[0]+params[1]*(1-np.exp(-params[5]*t))+params[2]*P+params[3]*np.log(P)+params[4]*P**2