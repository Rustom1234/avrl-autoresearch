def equation(t,P,params):
    return params[0]+params[1]*P+params[2]*np.log(P)+params[3]*np.sin(params[4]*t+params[5])*P**params[6]+params[7]*t