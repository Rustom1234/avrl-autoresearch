def equation(t,A,params):
    return -params[0]*t**(-params[1])/np.sqrt(A) + params[2]*A**params[3]*np.exp(-params[4]*t**params[5])+params[6]/np.sqrt(A)