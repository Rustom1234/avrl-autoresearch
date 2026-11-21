def equation(t,A,params):
    return -(params[0]/t+params[1]*np.log(t)+params[2]+params[3]*A**params[4]*t**params[5]+params[6]*A)