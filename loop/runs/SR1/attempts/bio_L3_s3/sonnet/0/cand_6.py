def equation(t,P,params):
    return params[0]+params[1]*P+params[2]*np.sqrt(P)+ (params[3]+params[4]*P)*np.exp(-params[5]*t)+params[6]*np.sin(params[7]*t+params[8])