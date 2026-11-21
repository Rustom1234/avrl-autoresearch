def equation(t,P,params):
    return params[0]*P*(P/params[1]-1)*(1-P/params[2])/(1+params[3]*P)+params[4]*np.sin(params[5]*t)