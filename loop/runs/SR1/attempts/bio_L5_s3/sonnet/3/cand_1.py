def equation(t,P,params):
    return params[0]*P*(P/params[1]-1)*(1-P/params[2])+params[3]*np.sin(params[4]*t)