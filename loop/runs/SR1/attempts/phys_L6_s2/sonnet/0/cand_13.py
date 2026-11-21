def equation(t,x,v,params):
    return t**params[0]*(params[1]+params[2]*v**params[3]+params[4]*x**params[5])+params[6]*np.sin(params[7]*t+params[8])