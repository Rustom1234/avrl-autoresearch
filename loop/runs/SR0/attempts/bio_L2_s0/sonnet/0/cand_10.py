def equation(t,P,params):
    return params[0]*P*np.log(params[1]/P) + params[2]*np.log(t+params[3])+params[4]*t+params[5]