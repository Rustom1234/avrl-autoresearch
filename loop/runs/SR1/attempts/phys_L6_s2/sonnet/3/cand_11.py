def equation(t,x,v,params):
    return params[0]*t**params[1]*v**params[2]+params[3]*x**params[4]+params[5]+params[6]*v**params[7]