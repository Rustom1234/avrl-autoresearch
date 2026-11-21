def equation(t,P,params):
    return params[0]+params[1]*t**params[2]+params[3]*P**params[4]+params[5]*np.log(P)+params[6]*P*t**params[2]