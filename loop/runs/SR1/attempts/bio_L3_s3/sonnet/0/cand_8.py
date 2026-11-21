def equation(t,P,params):
    g=1-np.exp(-params[4]*t)
    return params[0]+params[1]*P+params[2]*np.sqrt(P)+params[3]*g*(1+params[5]*P)+params[6]*np.log(P)+params[7]*P**2