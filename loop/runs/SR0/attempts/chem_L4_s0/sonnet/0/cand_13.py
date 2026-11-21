def equation(t, A, params):
    return (params[0]-params[1]/A+params[6]*A**2)*(1-np.exp(-params[2]*t))+params[3]*A+params[4]*A*t+params[5]*t