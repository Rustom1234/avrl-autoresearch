def equation(t,P,params):
    return -params[0]*(1-np.exp(-params[1]*t))*(1+params[2]*P+params[3]*P**2) + params[4]*P*(1-P/params[5]) + params[6]*P*t