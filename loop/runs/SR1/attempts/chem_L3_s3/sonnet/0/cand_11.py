def equation(t,A,params):
    return params[0]/t**params[1]*(1+params[2]*A)+params[3]+params[4]*A**2+params[5]*np.exp(-params[6]*t)