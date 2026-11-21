def equation(t,x,v,params):
    return x*v+params[0]+params[1]*t**params[2]+params[3]*np.exp(-params[4]*x)+params[5]*v**params[6]