def equation(t,A,params):
    return params[0]/t+params[1]+params[2]*np.log(t)+params[3]*A**2*(1+params[4]*np.log(t))+params[5]*A*np.sqrt(t)+params[6]*t