def equation(t,P,params):
    return params[0]+params[1]*np.log(P)+params[2]*P+params[3]*P**2+t*(params[4]+params[5]*P+params[6]*np.log(P))+params[7]*t**2*P+params[8]*t**2