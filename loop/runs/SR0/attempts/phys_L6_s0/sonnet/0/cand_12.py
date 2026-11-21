def equation(t,x,v,params):
    return params[0]+params[1]*t**params[2]+params[3]*np.log(x)+params[4]*np.cos(x)+params[5]*x*v+params[6]*v+params[7]*v**params[8]