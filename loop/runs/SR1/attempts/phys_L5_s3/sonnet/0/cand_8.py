def equation(t,x,v,params):
    return params[0]+params[1]*np.sin(x)+params[2]*np.cos(x)+params[3]*t+params[4]*t**2+params[5]*x**-0.5+params[6]*v+params[7]*x**-2