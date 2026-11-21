def equation(t,x,v,params):
    return params[0]/t**params[6]+params[1]*x**params[2]+params[3]*v**params[4]+params[5]*x*v+params[7]+params[8]*np.cos(t+params[9])