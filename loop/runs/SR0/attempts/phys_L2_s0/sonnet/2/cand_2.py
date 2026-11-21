def equation(t,x,v,params):
    return params[0]+params[1]*x+params[2]*x**3+params[3]*v+params[4]*v**2+params[5]*np.cos(params[6]*t+params[7])