def equation(t,A,params):
    return params[0]/t+params[1]*A/t+params[2]+params[3]*A**2+params[4]*np.exp(-params[5]*t)+params[6]*A