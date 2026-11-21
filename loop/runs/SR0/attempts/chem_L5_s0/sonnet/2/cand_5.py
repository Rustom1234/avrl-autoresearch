def equation(t,A,params):
    return params[0]*A*np.tanh(params[1]*A**params[2])+params[3]*A+params[4]*A**2