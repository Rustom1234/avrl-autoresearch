def equation(t,A,params):
    return -(params[0]*t**(1/3)+params[1]*t**-1.5+params[2]*A**2*np.sqrt(t)+params[3]+params[4]*np.log(t))