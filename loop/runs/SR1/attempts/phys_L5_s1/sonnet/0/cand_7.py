def equation(t,x,v,params):
    return -params[0]*np.sqrt(v)+params[1]*np.exp(-params[2]*t)+params[3]-params[4]*x+params[5]*np.cos(params[6]*t)