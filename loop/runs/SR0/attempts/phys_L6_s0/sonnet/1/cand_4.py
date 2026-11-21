def equation(t,x,v,params):
    return params[0]+params[1]*np.sqrt(t)+params[2]*np.log(v)+params[3]*x**params[4]+params[5]*x*v+params[6]*x