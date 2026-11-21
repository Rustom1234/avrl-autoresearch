def equation(t,x,v,params):
    return params[0]+params[1]*t+params[2]*x+params[3]*x*v+params[4]*np.sin(params[5]*v+params[6])+params[7]*np.log(v)+params[8]*np.cos(params[9]*t)