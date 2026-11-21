def equation(t,x,v,params):
    return params[0]*np.cos(0.7*t+params[1])+params[2]+params[3]*x**params[4]+params[5]*v**params[6]