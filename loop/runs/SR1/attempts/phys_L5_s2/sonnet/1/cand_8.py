def equation(t,x,v,params):
    return params[0]+params[1]*np.sin(params[2]*t+params[3])+params[4]*np.sin(params[5]*x)+params[6]*np.exp(-params[7]*v)