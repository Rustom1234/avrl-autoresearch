def equation(t,x,v,params):
    return params[0]*np.cos(0.7*t+params[1])+params[2]+params[3]*np.log(x)+params[5]*np.log(v)