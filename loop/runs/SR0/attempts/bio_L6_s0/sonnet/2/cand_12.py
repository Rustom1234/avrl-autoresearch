def equation(t,P,params):
    return params[0]*t+params[1]*P+params[2]*t*P**params[3]+params[4]*np.log(t)+params[5]*np.sqrt(t)