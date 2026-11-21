def equation(t,P,params):
    return -params[0]*(1-np.exp(-params[1]*t))*np.abs(P)**params[2] + params[3]*P*(1-P/params[4])