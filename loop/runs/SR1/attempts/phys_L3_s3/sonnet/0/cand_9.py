def equation(t,x,v,params):
    return params[0]+params[1]*x**params[2]+params[3]*v**params[4]*np.exp(-params[5]*t)+params[6]*t+params[7]*np.cos(params[8]*t)