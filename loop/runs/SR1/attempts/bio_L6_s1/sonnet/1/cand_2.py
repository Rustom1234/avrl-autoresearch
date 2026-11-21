def equation(t,P,params):
    return params[0]+params[1]*t+params[2]*np.log(P)+params[3]*P+params[4]*P*t+params[5]*P*np.log(P)+params[6]*t*np.log(P)