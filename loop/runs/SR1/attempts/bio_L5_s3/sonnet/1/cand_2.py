def equation(t,P,params):
    return params[0]+params[1]*np.log(P)+params[2]*P+params[3]*P**2+params[4]*P*np.log(P)