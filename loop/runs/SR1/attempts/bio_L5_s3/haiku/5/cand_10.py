def equation(t,P,params):
    return params[0]*P*(1-P/params[1])*(P/params[2]-1)+params[3]*P*np.log(P)+params[4]*P