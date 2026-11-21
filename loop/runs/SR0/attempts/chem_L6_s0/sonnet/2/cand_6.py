def equation(t,A,params):
    return params[0]*A**2*np.exp(-params[1]*t)+params[2]*A*t**params[3]+params[4]*t**params[5]