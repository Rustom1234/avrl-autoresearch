def equation(t,P,params):
    return params[0]+params[1]*P+params[2]*P**(-params[3])+(params[4]+params[5]*P+params[7]*P**2)*t**params[6]+params[8]*np.log(P)
