def equation(t,x,v,params):
    return params[0]*np.cos(params[1]*t+params[2])+params[3]+(params[5]+params[6]*t+params[7]*t**2)*np.abs(v)**params[4]