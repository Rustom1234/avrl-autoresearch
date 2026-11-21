def equation(t, A, params):
    return -params[0]+params[1]*A**2/(params[2]+A**2)*np.log(t+params[3])+params[4]*A*(1-params[5]*np.exp(-params[6]*t))