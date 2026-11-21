def equation(t,x,v,params):
    return params[0]*np.log(t)+params[1]+params[3]*x**params[4]+params[5]*v**params[6]