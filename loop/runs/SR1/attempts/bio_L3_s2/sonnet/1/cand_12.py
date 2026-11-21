def equation(t,P,params):
    return params[0]+params[1]*np.log(t+params[2])+P*(params[3]-np.sin(params[6]*t))+params[4]*P**params[5]