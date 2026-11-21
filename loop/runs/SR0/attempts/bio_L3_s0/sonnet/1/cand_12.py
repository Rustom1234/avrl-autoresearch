def equation(t,P,params):
    return params[0]+params[1]*P+params[2]*P**2+params[3]*np.log(P)+t*(params[4]*P**(-params[5])+params[6])+params[7]*t*P