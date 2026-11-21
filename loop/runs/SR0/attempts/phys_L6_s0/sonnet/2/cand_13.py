def equation(t,x,v,params):
    return params[0]+params[1]*np.sqrt(t)+params[2]*np.sin(params[3]*x+params[4])+params[5]*x+params[6]*x**1.5*v**2