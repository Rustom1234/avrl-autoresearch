def equation(t,A,params):
    return -(params[0]+params[1]*np.exp(-params[2]*t)+params[5]*t)/np.sqrt(A) + params[3]*A**3*np.exp(-params[4]*t*1.0)