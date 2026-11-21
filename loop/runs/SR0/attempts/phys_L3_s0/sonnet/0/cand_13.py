def equation(t,x,v,params):
    return t**params[0]*(params[1]*np.log(x)+params[2]+params[3]*x)+params[4]*v+params[5]+params[6]*x