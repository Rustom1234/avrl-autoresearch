def equation(t,A,params):
    return params[0]*np.exp(-params[1]*A**params[2])-params[3]*t-params[4]*A*np.sin(params[5]*t)