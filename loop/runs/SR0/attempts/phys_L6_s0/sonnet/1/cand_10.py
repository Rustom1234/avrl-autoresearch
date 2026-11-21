def equation(t,x,v,params):
    return params[0]+params[1]*t**params[2]+params[3]*x**params[4]+params[5]*x**params[6]*v**params[7]