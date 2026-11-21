def equation(t,P,params):
    return -params[0]*np.tanh(params[1]*t)*(1+params[2]*P+params[3]*P**2)+params[4]*t**params[5]*P**params[6]