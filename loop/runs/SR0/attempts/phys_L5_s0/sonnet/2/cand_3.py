def equation(t,x,v,params):
    return params[0]*np.cos(params[1]*t) - params[3]*x - params[4]*x**3 - params[5]*v*np.abs(v)-params[6]*v + params[7]