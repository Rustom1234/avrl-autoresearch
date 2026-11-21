def equation(t,x,v,params):
    return params[0]+params[1]*np.sin(params[2]*t+params[3])+params[4]*x+params[5]*v+params[6]*x**2+params[7]*v**2+params[8]*x*v