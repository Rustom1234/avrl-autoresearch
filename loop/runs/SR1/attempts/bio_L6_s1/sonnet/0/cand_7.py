def equation(t,P,params):
    return params[0]*np.exp(-params[1]*P)+params[2]*P+params[3]+(params[4]+params[5]*P)*t**params[6]+params[7]*P**2