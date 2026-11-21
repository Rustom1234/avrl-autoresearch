def equation(t,P,params):
    return params[0]*np.log(1+params[1]*P)-params[2]*P**params[3]