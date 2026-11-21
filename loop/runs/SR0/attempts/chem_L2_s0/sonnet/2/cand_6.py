def equation(t,A,params):
    return -(params[0]+params[1]*t)/np.sqrt(A)*1.0 + params[2]*A**params[3]*np.exp(-params[4]*t)+params[5]*np.exp(-params[6]*t)/np.sqrt(A)