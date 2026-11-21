def equation(t,x,v,params):
    return params[0]*np.cos(params[1]*t+params[2])+params[3]+params[4]*x+params[5]*np.exp(-params[6]*v)+params[7]*x**2