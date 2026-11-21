def equation(t,P,params):
    return -params[0]*P**1.5*np.exp(params[1]*t) + params[2]*P+params[3]*P**2