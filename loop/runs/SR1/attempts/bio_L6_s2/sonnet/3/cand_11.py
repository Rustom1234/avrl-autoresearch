def equation(t,P,params):
    return -P*(params[0]+params[1]*np.log(t)+params[2]*P+params[3]*P*np.log(t))