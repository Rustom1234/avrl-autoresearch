def equation(t,A,params):
    return -(params[0]*np.exp(-params[1]*t)+params[2]+params[3]*A**params[4]*t**params[5])