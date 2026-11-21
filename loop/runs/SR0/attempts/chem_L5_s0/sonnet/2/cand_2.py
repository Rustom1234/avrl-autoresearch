def equation(t,A,params):
    return params[0]*A+params[1]*A**2+params[2]*np.sin(params[3]*A)+params[4]*np.exp(-params[5]*A)