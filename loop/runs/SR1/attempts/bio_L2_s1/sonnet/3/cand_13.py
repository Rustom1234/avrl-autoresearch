def equation(t,P,params):
    return params[0]+params[1]*np.log(t+params[2])+params[3]*P+params[4]*P**2+params[5]*P*np.log(t+params[2])+params[6]*P**3