def equation(t,x,v,params):
    return params[0]*x**params[1]*t**params[2]+params[3]*v**params[4]*t**params[5]+params[6]*t**params[7]+params[8]