def equation(t,x,v,params):
    return -params[0]*x**params[1]+params[2]*x**2/v+params[3]*np.exp(-params[4]*t)+params[5]*np.sin(params[6]*t+params[7])+params[8]*v