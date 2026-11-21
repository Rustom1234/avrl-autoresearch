def equation(t,P,params):
    return params[0]/P+params[2]*P+params[4]+params[5]*np.sqrt(t)+params[6]*P*t