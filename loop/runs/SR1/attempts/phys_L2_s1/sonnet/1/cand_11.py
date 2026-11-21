def equation(t,x,v,params):
    return params[0]*x+params[1]*x**3+params[2]*v+params[3]*np.sin(params[4]*x)+params[5]*np.exp(-params[6]*v)+params[7]*x*v+params[8]