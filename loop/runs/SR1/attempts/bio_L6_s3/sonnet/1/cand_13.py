def equation(t,P,params):
    return -params[0]*(1-np.exp(-params[1]*t*P**params[2])) + params[3]*P*(1-P/params[4]) + params[5]*P*t