def equation(t,x,v,params):
    return params[0]+(params[1]+params[2]*t)*np.sin(params[3]*t+params[4])+params[5]*x+params[6]*v