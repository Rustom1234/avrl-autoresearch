def equation(t,x,v,params):
    return params[0]+params[1]*np.sqrt(t)+params[3]*np.log(v+params[4])+params[5]*t**1.5/x