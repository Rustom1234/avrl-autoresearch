def equation(t,A,params):
    return params[0]*np.exp(-params[1]*A)+params[2]*np.exp(-params[3]*A**2)+params[4]*A+params[5]