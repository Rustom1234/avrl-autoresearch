def equation(t,x,v,params):
    return params[0]+params[1]*t+params[2]*x+params[3]*x**3+params[5]*v+params[6]*v**2+params[7]*np.sin(params[8]*t)