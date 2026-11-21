def equation(t,A,params):
    return -params[0]*(1+params[1]*np.log(t))*A*np.exp(-params[2]*A)+params[3]*A