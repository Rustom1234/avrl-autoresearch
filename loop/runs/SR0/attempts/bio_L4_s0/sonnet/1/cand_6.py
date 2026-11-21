def equation(t,P,params):
    return params[0]+params[1]*P/(1+params[2]*P)+params[3]*t+params[4]*P**params[5]*np.exp(-params[6]*t)