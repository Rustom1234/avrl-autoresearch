def equation(t,A,params):
    return params[0]*A/(params[1]+A)*np.exp(-params[2]*t)+params[3]*A**params[4]+params[5]*t