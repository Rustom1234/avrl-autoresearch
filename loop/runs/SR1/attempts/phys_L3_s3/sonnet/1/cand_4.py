def equation(t,x,v,params):
    return params[0]+params[1]*v**params[2]+params[3]*t+params[4]*x+params[5]*x**2+params[6]*np.cos(params[7]*t+params[8])