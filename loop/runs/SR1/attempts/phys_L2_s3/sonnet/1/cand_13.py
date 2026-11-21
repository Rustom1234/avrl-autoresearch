def equation(t,x,v,params):
    return params[0]*t/x+params[1]*x**3+params[2]*np.sqrt(t)+params[3]+params[4]*np.sin(params[5]*t+params[6])