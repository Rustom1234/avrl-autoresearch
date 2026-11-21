def equation(t,x,v,params):
    return params[0]*np.cos(params[1]*t+params[2])+params[3]*np.abs(x)**params[4]+params[5]*np.abs(v)**params[6]+params[7]