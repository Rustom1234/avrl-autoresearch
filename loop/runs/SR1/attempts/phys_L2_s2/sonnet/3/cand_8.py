def equation(t,x,v,params):
    return params[0]+params[1]*x+params[2]*np.log(x)+params[3]/x+params[4]*x/v