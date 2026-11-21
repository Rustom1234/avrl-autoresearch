def equation(t,x,v,params):
    return params[0]*np.cos(params[1]*t) - params[3]*x**params[2] - params[5]*v**params[6] + params[7]