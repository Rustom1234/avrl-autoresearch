def equation(t,x,v,params):
    return (params[0]*t**params[1]+params[5]*v**params[2]+params[3])*x**params[4]+params[6]*x**params[7]