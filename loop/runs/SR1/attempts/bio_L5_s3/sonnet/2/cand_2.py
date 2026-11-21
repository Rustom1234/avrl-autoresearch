def equation(t,P,params):
    return params[0]*P*np.log(params[1]/P)-params[2]*P/(params[3]+P)+params[4]+params[5]*np.sin(params[6]*t)