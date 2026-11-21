def equation(t,x,v,params):
    return params[7]*x*v+params[0]+params[1]*t**params[2]+params[3]*x**params[4]+params[5]*v**params[6]