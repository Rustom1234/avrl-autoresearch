def equation(t,x,v,params):
    return params[0]+params[1]*x**params[2]+params[3]*v**params[4]+ (params[5]+params[6]*x+params[7]*v)*t**params[8]