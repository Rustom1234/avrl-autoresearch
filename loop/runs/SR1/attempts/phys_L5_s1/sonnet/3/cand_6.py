def equation(t,x,v,params):
    return params[0]+params[1]*np.sqrt(v)+params[2]*np.log(t)+params[3]*x+params[4]*x**3