def equation(t,x,v,params):
    return params[0]+params[1]*np.log(x)+v*(params[2]+params[3]*np.log(x))+params[4]*np.cos(params[5]*t+params[6])