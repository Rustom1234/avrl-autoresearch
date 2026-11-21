def equation(t,x,v,params):
    return params[0]*x*np.log(t+params[1])+params[2]*v+params[3]/x+params[4]*x+params[5]+params[6]*np.sin(params[7]*t)