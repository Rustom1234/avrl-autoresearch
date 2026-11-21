def equation(t,P,params):
    return -params[0]*np.tanh(params[1]*t*P**params[2]) + params[3]*P**2*np.sin(params[4]*t+params[5])*0+params[6]*P**params[7]*t**params[8]