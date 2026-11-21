def equation(t,A,params):
    g=params[0]*t**(-params[1])+params[2]*t*0
    return -(g - params[3]*A**params[4]*np.exp(-params[5]*t))/np.sqrt(A)