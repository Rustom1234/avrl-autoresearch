def equation(t,x,v,params):
    w=params[2]
    return params[0]*np.cos(w*t+params[3])+params[4]-params[5]*v+params[6]*v**2-params[7]*x+params[8]*v*np.cos(w*t+params[9])