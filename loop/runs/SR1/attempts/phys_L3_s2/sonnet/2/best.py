def equation(t,x,v,params):
    return params[0]*x+params[1]*x**3+params[2]*v+params[3]*v*x**2+params[4]*np.cos(params[5]*t+params[6])*(1+params[7]*x)+params[8]