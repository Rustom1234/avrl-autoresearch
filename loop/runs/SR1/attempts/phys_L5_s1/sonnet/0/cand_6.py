def equation(t,x,v,params):
    return -params[0]*v**params[1]+params[2]/(t+params[3])**params[4]+params[5]-params[6]*x