def equation(t,P,params):
    return params[0]+params[1]*P+params[2]*P*P+params[3]*np.log(P)+params[4]*np.sin(params[5]*t)