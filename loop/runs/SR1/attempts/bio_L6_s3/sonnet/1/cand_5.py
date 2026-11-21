def equation(t,P,params):
    return -params[0]*(1-np.exp(-params[1]*t)) + params[2]*P + params[3]*P**2 + params[4]*P**2*t+params[5]*P*t