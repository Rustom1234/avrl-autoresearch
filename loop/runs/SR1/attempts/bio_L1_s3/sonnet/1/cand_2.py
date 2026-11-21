def equation(t,P,params):
    return params[0]*np.log(params[1]*P+1)*(1-P/params[2])+params[3]