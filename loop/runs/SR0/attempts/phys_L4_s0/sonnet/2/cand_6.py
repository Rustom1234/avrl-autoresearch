def equation(t,x,v,params):
    return params[0]+params[1]*np.cos(params[2]*t+params[3])+params[4]*x+params[5]*x**3+params[6]*v+params[7]*np.abs(v)**params[8]*v/np.abs(v)