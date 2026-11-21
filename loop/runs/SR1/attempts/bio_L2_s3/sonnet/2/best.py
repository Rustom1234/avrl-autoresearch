def equation(t,P,params):
    return params[0]*np.exp(-params[1]*P)+params[2]*t*(1+params[3]*P)+params[4]*np.sin(params[5]*t+params[6])+params[7]