def equation(t,P,params):
    return -params[0]*(1-np.exp(-params[1]*t)) + params[2]*P + params[3]*P**2 + params[4]*t*P + params[5]*t*P**2 + params[6]*t**2*P