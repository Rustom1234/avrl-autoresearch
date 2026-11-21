def equation(t,A,params):
    return params[0]*A**params[1]*(1+params[2]*np.log(t+params[3]))-params[4]-params[5]*t**2*A**params[6]