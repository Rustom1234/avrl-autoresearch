def equation(t,P,params):
    return params[0]*P/(params[1]+P)+t*params[2]*np.exp(-params[3]*P)+params[4]*t*P+params[5]*t**2*P+params[6]*t**2*np.exp(-params[7]*P)