def equation(t,x,v,params):
    return params[0]*np.cos(params[1]*t+params[2])*x**params[3]*v**params[4] + params[5]*x**params[6] + params[7]*v**params[8]+params[9]