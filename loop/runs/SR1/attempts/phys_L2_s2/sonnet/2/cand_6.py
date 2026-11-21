def equation(t,x,v,params):
    return params[0]+params[1]*x+params[2]*x**2+params[3]/x+params[4]*x**params[5]/v**params[6]+params[7]*np.sin(t)