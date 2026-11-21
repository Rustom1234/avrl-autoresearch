def equation(t,x,v,params):
    return (params[0]+params[1]*x)*v**params[2]+params[3]*x**params[4]+params[5]*v**params[6]+params[7]+params[8]*x**2