def equation(t,P,params):
    return params[0]*P*np.log(params[1]/P)-params[2]/P+params[3]