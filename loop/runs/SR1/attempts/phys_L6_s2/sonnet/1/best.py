def equation(t,x,v,params):
    return params[0]+params[1]*t**params[2]*x**params[3]+params[4]*v**params[5]+params[6]*x**params[7]