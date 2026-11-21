def equation(t,P,params):
    return params[0]+params[1]*np.log(P)+params[2]*t+params[3]*np.exp(params[4]*t)*P**params[5]