def equation(t,x,v,params):
    return params[0]*t**params[1]*x**params[2]+params[3]*v**params[4]*x**params[5]+params[6]*x**params[7]+params[8]