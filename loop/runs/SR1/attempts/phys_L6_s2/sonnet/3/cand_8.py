def equation(t,x,v,params):
    return params[0]*t**params[1]+params[2]*x**params[3]+params[4]*v**params[5]+params[6]+params[7]*np.sin(params[8]*t+params[9])