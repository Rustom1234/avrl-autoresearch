def equation(t,P,params):
    return params[0]+params[1]*P**(params[2]+params[3]*np.log(t))+params[4]*np.exp(-params[5]*t)+params[6]*np.log(t)