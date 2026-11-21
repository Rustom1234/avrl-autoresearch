def equation(t,P,params):
    return params[0]*P/(1+params[1]*P)+params[2]+params[3]*t*np.exp(-t/params[4])