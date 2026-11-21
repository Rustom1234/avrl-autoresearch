def equation(t,P,params):
    return params[0]+params[1]*t+params[2]*P*np.log(P)+params[3]*P+params[4]*P**2+params[5]*t*P+params[6]*np.log(P)