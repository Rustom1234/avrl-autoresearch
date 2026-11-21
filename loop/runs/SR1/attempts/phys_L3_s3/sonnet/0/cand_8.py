def equation(t,x,v,params):
    return params[0]+params[1]*x**params[2]+params[3]*np.log(v)+params[4]*t+params[5]*np.sin(params[6]*t+params[7])+params[8]*v