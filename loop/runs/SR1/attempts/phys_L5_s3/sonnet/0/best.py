def equation(t,x,v,params):
    return params[0]-params[1]*np.exp(-params[2]*x)+params[3]*x+params[4]*x**2+params[5]*x**3+params[6]*t+params[7]*t**2+params[8]*t**3