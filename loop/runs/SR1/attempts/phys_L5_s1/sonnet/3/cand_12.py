def equation(t,x,v,params):
    return params[0]+params[1]*np.sqrt(v)+params[2]*v+params[3]*np.cos(0.6*params[4]*t)+params[5]*np.sin(0.6*params[4]*t)+params[6]*x