def equation(t,x,v,params):
    return params[0]*t/x+params[1]*x**3+params[2]*np.sqrt(t)+params[3]+params[4]*v+params[5]*x+params[6]*v**2