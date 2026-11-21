def equation(t,A,params):
    return params[0]*A*(1+params[1]*np.log(t))-params[2]+params[3]*np.sin(params[4]*t)*A