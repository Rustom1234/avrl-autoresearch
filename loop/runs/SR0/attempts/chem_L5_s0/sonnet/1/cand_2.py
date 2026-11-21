def equation(t,A,params):
    return params[0]*np.sin(params[1]*A+params[2]*t+params[3])+params[4]