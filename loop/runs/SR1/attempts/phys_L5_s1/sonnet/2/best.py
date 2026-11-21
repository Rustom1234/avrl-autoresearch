def equation(t,x,v,params):
    return params[0]*np.exp(-params[1]*t)-params[2]*v**params[3]+params[4]*x+params[5]*x**3