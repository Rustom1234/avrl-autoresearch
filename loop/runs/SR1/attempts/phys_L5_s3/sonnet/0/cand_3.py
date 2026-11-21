def equation(t,x,v,params):
    return params[0]*np.tanh(params[1]*x)+params[2]*t+params[3]*v+params[4]*np.cos(params[5]*t+params[6])+params[7]