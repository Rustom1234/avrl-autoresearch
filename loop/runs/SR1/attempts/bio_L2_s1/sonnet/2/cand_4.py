def equation(t,P,params):
    return params[0]+params[1]*t+params[2]*P+params[3]*P**2+params[4]*np.log(P)+params[5]*t*P+params[6]/P