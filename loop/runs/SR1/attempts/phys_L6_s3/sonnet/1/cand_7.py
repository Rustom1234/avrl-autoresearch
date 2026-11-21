def equation(t,x,v,params):
    return params[0]*np.cos(params[1]*t+params[2]) + params[3]*x**params[4] + params[5]*v**params[6] + params[7]*x**params[8]*v**params[9]