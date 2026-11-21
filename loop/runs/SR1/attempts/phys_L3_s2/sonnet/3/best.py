def equation(t,x,v,params):
    return params[0]+params[1]*v+params[2]*t**params[3]+params[4]*np.log(x)+params[5]*v**2+params[6]*x*v+params[7]*np.cos(t)+params[8]*np.sin(t)