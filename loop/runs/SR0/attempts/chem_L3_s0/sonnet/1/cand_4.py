def equation(t,A,params):
    return params[0]*A*np.log(params[1]*t+params[2])-params[3]+params[4]*A**2