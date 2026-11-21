def equation(t,x,v,params):
    return params[0]+params[1]*x+params[2]*x**2+params[3]*v+np.sin(params[4]*t+params[5])*(params[6]+params[7]*v+params[8]*x)+params[9]*v**2