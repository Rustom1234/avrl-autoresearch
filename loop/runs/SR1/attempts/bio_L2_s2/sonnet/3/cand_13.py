def equation(t,P,params):
    return params[0]*P+params[1]*P*np.log(P)+params[2]+params[3]*P*P+params[4]*t*P+params[5]*t+params[6]*t*t+params[7]*t*t*P