def equation(t,P,params):
    return params[0]+params[1]*np.log(P)+params[2]*t+params[3]*t*P+params[4]*np.sin(params[5]*t+params[6])