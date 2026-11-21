def equation(t,P,params):
    return params[0]+params[1]*P*(1-P/params[2]) + params[3]*np.log(t+params[4])+params[5]*P*np.log(t+params[4])