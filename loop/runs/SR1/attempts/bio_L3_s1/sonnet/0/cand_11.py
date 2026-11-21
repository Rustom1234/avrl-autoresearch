def equation(t,P,params):
    return params[0]/P+params[1]*P**2+P*(params[2]+params[3]*t)+params[4]*t+params[5]+params[6]*P*np.sin(params[7]*t)