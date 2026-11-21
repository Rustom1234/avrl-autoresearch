def equation(t,P,params):
    return params[0]*P+params[1]*P**2+params[2]*np.log(P)+params[3]+params[4]*np.sin(params[5]*t+params[6])+params[7]*t