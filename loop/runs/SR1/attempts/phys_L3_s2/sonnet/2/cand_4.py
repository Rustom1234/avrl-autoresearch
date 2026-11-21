def equation(t,x,v,params):
    return params[0]+params[1]*x+params[2]*v**params[3]+params[4]*np.sin(params[5]*t+params[6])