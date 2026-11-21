def equation(t,x,v,params):
    return -9.8+params[0]*x+params[1]/x+params[2]*x**params[3]+params[4]*x**params[5]/v**params[6]