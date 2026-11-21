def equation(t,x,v,params):
    return params[0]+params[1]*v**params[2]+params[3]*np.cos(params[4]*t+params[5]*0.1)+params[6]*x