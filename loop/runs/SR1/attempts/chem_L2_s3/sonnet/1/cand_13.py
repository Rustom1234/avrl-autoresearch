def equation(t,A,params):
    return -(params[0]*A**2*np.sqrt(t)+params[1]*np.exp(-params[2]*t)+params[3]/t+params[7]+params[4]*np.sqrt(t))