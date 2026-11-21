def equation(t,A,params):
    return -(params[0]*A**params[1] + params[2]*np.exp(params[3]*t)*np.exp(-params[4]*A))