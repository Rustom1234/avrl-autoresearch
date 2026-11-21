def equation(t,P,params):
    return params[0]/P + params[1]*P + params[2] + params[3]*np.sin(params[5]*t)*P*(1-P/params[4])+params[6]*P**2