def equation(t,P,params):
    return -params[0]*(1-np.exp(-params[1]*t)) - params[2]*P + params[3]*P**2 + params[4]*np.sin(params[5]*t+params[6]) + params[7]*P*np.exp(-params[8]*t)