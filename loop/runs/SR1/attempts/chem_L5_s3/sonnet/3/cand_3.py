def equation(t, A, params):
    return params[0]*A/(params[1]+A)*np.exp(-params[2]*t)*(1+params[3]*t)+params[4]*A**params[5]*np.exp(-params[6]*t)+params[7]