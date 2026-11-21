def equation(t,P,params):
    return params[0]+params[1]*np.log(P)+params[2]*P**params[3]+params[4]*t/P+params[5]*t