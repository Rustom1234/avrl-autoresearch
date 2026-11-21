def equation(t,P,params):
    return params[0]+params[1]*P+params[2]*np.exp(-params[3]*P)+params[4]*t**2*np.sqrt(P)+params[5]*P**2+params[6]*t+params[7]*np.log(P)