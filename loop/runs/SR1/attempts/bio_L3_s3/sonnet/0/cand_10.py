def equation(t,P,params):
    return params[0]+params[1]*P+params[2]*np.sqrt(P)+params[3]*np.sin(params[4]*t/3+params[5])+params[6]*P*np.sin(params[4]*t/3+params[5])