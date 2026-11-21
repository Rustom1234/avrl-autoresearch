def equation(t,x,v,params):
    return params[0]+params[1]*v**params[2]+params[3]*np.sin(params[4]*x)+params[5]*t**params[6]+params[7]*np.cos(params[8]*t)