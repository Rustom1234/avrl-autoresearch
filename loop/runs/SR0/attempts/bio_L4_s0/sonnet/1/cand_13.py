def equation(t,P,params):
    return params[0]+params[1]*P**params[2]+(params[3]*P+params[4])*np.exp(-params[5]*t**params[7])+params[6]*t