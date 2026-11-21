def equation(t,x,v,params):
    return params[0]+params[1]*np.sqrt(t)+params[2]*x**params[3]+params[4]*x*v**2+params[5]*v+params[6]*v**2+params[7]*x