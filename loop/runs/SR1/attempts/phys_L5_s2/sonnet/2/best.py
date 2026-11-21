def equation(t,x,v,params):
    return params[0]*np.sin(params[1]*t)+params[2]+params[3]*t+params[4]*t**2