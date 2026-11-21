def equation(t,P,params):
    return params[0]*np.exp(-params[1]*P)+params[2]+params[3]*P**2+params[4]*np.log(t)+params[5]*np.sin(params[6]*t+params[7])