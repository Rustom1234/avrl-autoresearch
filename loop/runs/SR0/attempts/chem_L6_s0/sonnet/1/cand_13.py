def equation(t,A,params):
    return params[0]*t**params[3]+params[1]*A**params[4]+params[2]*A**params[5]*np.exp(-params[6]*t)