def equation(t,x,v,params):
    return params[0]+params[1]*np.sqrt(v)+params[2]*np.log(t)+params[3]*t+params[4]*np.sqrt(t)+params[5]*x*np.sqrt(v)+params[6]*x