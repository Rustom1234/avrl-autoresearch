def equation(t,x,v,params):
    return params[0]+params[1]*np.sqrt(t)+params[2]*np.log(x)+params[3]*x**params[4]*v**params[5]+params[6]*x+params[7]*v