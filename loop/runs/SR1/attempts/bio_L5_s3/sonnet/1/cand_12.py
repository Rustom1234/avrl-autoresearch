def equation(t,P,params):
    return params[0]*np.log(P/params[1])+params[2]*P*(1-P/params[3])+params[4]*P