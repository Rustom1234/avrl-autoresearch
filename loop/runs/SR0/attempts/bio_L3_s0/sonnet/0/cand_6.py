def equation(t,P,params):
    return params[0]*P**(-params[1])*np.exp(params[2]*t)+params[3]*P+params[4]+params[5]*P*P