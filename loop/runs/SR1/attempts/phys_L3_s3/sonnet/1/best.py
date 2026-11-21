def equation(t,x,v,params):
    return params[0]+params[1]*np.log(v+params[2])+params[3]*t+params[4]*t**2+params[5]*x*np.log(v+params[2])+params[6]*x+params[7]*v