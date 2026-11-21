def equation(t,P,params):
    return params[0]+params[1]*np.log(P)+params[2]*np.exp(params[3]*P)+params[4]*t*P**(-params[5])