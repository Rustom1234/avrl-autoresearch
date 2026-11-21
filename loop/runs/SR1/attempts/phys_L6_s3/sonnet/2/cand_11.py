def equation(t,x,v,params):
    return params[0]+params[1]*x**params[2]+params[3]*v**params[4]+np.cos(0.5*t)*(params[5]+params[6]*x+params[7]*v)