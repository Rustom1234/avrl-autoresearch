def equation(t, A, params):
    return params[0]*A**params[1]*(1+params[2]*t)*np.exp(-params[3]*t)+params[4]*np.exp(-params[5]*t)+params[6]*np.log(A)