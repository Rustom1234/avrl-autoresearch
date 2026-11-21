def equation(t,x,v,params):
    return params[0]*v+params[1]*x+params[2]*x**3+params[3]*np.sin(params[4]*t+params[5])+params[6]*v**params[7]+params[8]