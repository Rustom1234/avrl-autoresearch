def equation(t,x,v,params):
    return params[0]*np.cos(params[1]*t+params[2])*v**params[3] + params[4]*np.log(x)+params[5]*x + params[6]*np.exp(-params[7]*v)+params[8]+params[9]*v