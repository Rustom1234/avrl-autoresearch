def equation(t,P,params):
    return params[0]/P+params[1]*P**params[2]+params[3]*np.log(P)*t+params[4]*t+params[5]*P*t+params[6]