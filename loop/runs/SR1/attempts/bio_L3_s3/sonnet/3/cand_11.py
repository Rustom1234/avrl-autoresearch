def equation(t,P,params):
    return params[0]+params[1]*P**params[6]+(params[2]+params[3]*P)*(1-np.exp(-t/params[4]))+params[5]*t*P/(1+t)