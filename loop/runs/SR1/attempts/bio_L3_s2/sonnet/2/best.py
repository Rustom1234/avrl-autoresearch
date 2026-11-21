def equation(t,P,params):
    return params[0]+params[1]*t+P*(params[2]+params[3]*np.sin(params[4]*t+params[5]))+params[6]*P**2+params[7]*t*P