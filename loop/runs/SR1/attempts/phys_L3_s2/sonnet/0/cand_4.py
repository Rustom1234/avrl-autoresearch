def equation(t,x,v,params):
    return params[0]+params[1]*v**params[2]+params[3]*x+params[4]*x**3+params[5]*t+params[6]*np.cos(t)