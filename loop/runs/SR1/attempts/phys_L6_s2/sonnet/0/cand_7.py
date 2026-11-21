def equation(t,x,v,params):
    return params[0]*t**params[1]*v**params[2]+params[3]*x**params[4]+params[5]*np.sin(params[6]*t)+params[7]