def equation(t, A, params):
    return params[0] + params[1]*A*(1+params[2]*np.log(t))*np.exp(-params[3]*A)+params[4]*A**2