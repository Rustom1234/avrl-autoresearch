def equation(t,A,params):
    return -(params[0]*A**params[1]*np.exp(-params[2]*t**params[3]) + params[4]*t**params[5]*np.exp(-params[6]*A))