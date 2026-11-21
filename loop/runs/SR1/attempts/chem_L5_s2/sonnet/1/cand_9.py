def equation(t, A, params):
    return params[0]+params[1]*t+params[2]*np.sqrt(t)+params[3]*np.exp(-params[4]*t)+params[5]*np.log(A)*np.exp(-params[6]*t)+params[7]*A