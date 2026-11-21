def equation(t,x,v,params):
    return params[0]*x+params[1]*x**params[2]+params[3]*v**params[4]+params[5]*x**params[6]*v**params[7]+params[8]