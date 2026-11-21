def equation(t,x,v,params):
    return params[0]+params[1]*np.log(x)+params[2]/v+params[3]*x+np.cos(0.5*t)*(params[4]+params[5]/v)+params[6]*v+params[7]*np.sin(0.5*t)