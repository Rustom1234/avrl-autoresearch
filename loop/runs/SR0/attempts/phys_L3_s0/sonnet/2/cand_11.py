def equation(t,x,v,params):
    return params[0]*x**params[1]*t**params[2]+params[3]*v+params[4]*x**params[5]+params[6]+params[7]*np.sin(t)+params[8]*np.cos(t)