def equation(t,x,v,params):
    return -params[0]*x**params[1]-params[2]*v**params[3]*x**params[4]+params[5]*np.exp(-params[6]*t)+params[7]*np.sin(params[8]*t)