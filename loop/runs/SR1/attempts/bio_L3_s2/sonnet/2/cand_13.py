def equation(t,P,params):
    return params[0]+params[1]*t+P*(params[2]+params[3]*np.cos(params[4]*t))+params[6]*P**2+params[7]*t*P+params[5]*np.sqrt(P)*t