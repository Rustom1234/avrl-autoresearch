def equation(t,A,params):
    return -(params[0]+params[1]*np.exp(-params[2]*t)+params[3]*np.log(t+params[8]) + A*(params[4]+params[5]*t**params[6]) + params[7]*A**2)