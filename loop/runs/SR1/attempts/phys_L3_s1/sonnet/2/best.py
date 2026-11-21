def equation(t,x,v,params):
    return params[0]*np.cos(params[1]*t+params[2])+params[3]+params[4]*np.exp(params[5]*t)*v**params[6]+params[7]*x