def equation(t,A,params):
    return -(params[0]*np.exp(-params[1]*t)+params[2]+params[3]*np.log(t)+params[6]*t+params[4]*A**2*t**params[5])