def equation(t,A,params):
    return params[0]*t+params[1]*A**2*(params[2]+np.exp(-params[3]*t))+params[4]*A*t