def equation(t,P,params):
    return params[0]*t*(1+params[1]*np.log(1+t))+params[2]*P+params[3]*P**2*(1+params[4]*t)+params[5]*t*P