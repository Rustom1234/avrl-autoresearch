def equation(t,P,params):
    return params[0]+params[1]*t+P*(params[2]+params[3]*t)+params[4]*P*P+params[5]*np.sin(params[6]*t+params[7])