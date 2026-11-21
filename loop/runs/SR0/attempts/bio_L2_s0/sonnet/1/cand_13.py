def equation(t,P,params):
    return params[0]*P*(1-P/params[1])*np.exp(-params[2]*t) - params[3]*np.tanh(params[4]*t) - params[5]*P*(1-np.exp(-params[6]*t))+params[7]*np.sin(params[8]*t)