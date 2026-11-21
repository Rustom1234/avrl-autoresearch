def equation(t,P,params):
    return params[0]/(P+params[1])+params[2]+params[3]*t*np.log(P+params[4])+params[5]*t