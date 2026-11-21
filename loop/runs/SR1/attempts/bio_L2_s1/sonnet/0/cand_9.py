def equation(t,P,params):
    return params[0]+params[1]*np.log(P)+params[2]*P/(1+params[3]*t)+params[4]*t