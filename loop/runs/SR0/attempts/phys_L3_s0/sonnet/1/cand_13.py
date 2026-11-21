def equation(t,x,v,params):
    return params[0]*x**params[1]+params[2]*v+params[3]*x*np.sqrt(t)+params[4]*np.sin(params[5]*t+params[6])+params[7]*t+params[8]