def equation(t,P,params):
    return params[0]*np.exp(-params[1]*P)+params[2]*P+params[3]+(params[4]+params[5]*P)*np.sin(params[6]*t+params[7])+params[8]*P**2