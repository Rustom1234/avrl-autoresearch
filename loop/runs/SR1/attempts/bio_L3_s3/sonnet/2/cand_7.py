def equation(t,P,params):
    return params[0]*P*(1-P/params[1])+params[2]+params[3]*np.sin(params[4]*t+params[5])