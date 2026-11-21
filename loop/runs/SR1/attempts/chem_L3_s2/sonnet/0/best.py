def equation(t, A, params):
    return params[0]-params[1]*np.exp(-params[2]*t)+params[6]*np.log(t)+A**3*(params[3]+params[4]*np.exp(-params[5]*t))